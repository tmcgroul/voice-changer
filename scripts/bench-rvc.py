#!/usr/bin/env python3
"""Chunked real-time RVC inference latency benchmark (SPEC §4.2).

Measures per-chunk inference cost through Applio's own inference internals
(C7: measure their code paths, do not reimplement the pipeline):

  f0:    rvc.infer.pipeline.Pipeline.get_f0, rmvpe branch
  embed: the exact `model(feats)["last_hidden_state"]` call from
         Pipeline.voice_conversion
  faiss: Pipeline._retrieve_speaker_embeddings (their method, including the
         MPS float32 workaround they already ship)
  gen:   the exact F.interpolate + net_g.infer tail from
         Pipeline.voice_conversion
  glue:  tensor conversions / p_len slicing between their stages

Only the timing instrumentation is local; every op is theirs, in their order.
Stages are synchronized (torch.mps.synchronize) so per-stage numbers are
honest and the stage sums account for ~all of the measured total.

Each timed chunk passes through the pipeline the same way a chunk of a short
file would: f0 over the context-padded audio, one voice_conversion call —
identical to Pipeline.pipeline's path for anything shorter than x_max (32 s
on MPS). Context pad defaults to their t_pad (= x_pad = 1.0 s per side, what
their whole-file path pads); --pad explores slimmer chunker contexts.

Example:
  external/Applio/.venv/bin/python scripts/bench-rvc.py \
      --model data/<voice>/<voice>_100e_2700s.pth --index data/<voice>/<voice>.index \
      --audio data/<voice>/selected/<clip>.wav \
      --windows 0.1,0.25,0.5,1.0 --reps 25 --json /tmp/bench.json
"""

import argparse
import json
import math
import os
import statistics
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APPLIO = os.path.join(REPO, "external", "Applio")

def _parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="fine-tuned .pth to bench")
    ap.add_argument("--index", required=True, help="matching faiss .index file")
    ap.add_argument("--audio", required=True,
                    help="a speech wav from the model's dataset (real chunks)")
    ap.add_argument("--windows", default="0.1,0.25,0.5,1.0")
    ap.add_argument("--reps", type=int, default=25, help="timed chunks per window size")
    ap.add_argument("--warmup", type=int, default=5, help="discarded chunks per window size")
    ap.add_argument("--index-rate", type=float, default=0.75)
    ap.add_argument("--pad", type=float, default=1.0,
                    help="context seconds padded per side (Applio's t_pad = 1.0)")
    ap.add_argument("--device", choices=("mps", "cpu"), default=None,
                    help="override device (default: Applio's inference_config)")
    ap.add_argument("--json", default=None, help="write full results to this path")
    return ap.parse_args()


ARGS = _parse_args()
if ARGS.json is not None:
    ARGS.json = os.path.abspath(ARGS.json)

# Applio's loaders resolve pretrained assets relative to cwd (now_dir = getcwd()).
os.chdir(APPLIO)
sys.path.insert(0, APPLIO)

import faiss  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from rvc.configs.config import inference_config  # noqa: E402
from rvc.infer.infer import VoiceConverter  # noqa: E402
from rvc.lib.utils import load_audio_infer  # noqa: E402

WINDOW = 160  # their self.window: 160 samples @16 kHz = 10 ms f0 frames
STAGES = ("f0", "embed", "faiss", "gen", "glue")


def pctl(xs, p):
    xs = sorted(xs)
    k = (len(xs) - 1) * p / 100.0
    f = math.floor(k)
    c = min(f + 1, len(xs) - 1)
    return xs[f] if f == c else xs[f] + (xs[c] - xs[f]) * (k - f)


def sync(dev):
    if dev == "mps":
        torch.mps.synchronize()


def run_chunk(pipeline, conv, audio_pad, sid_t, index_rate, protect=0.5):
    """One chunk through their stages with timers. Returns stage->ms dict."""
    dev = pipeline.device
    m = {}
    audio_f32 = audio_pad.astype(np.float32)
    p_len = audio_f32.shape[0] // WINDOW

    t0 = time.perf_counter()
    # f0 stage — their Pipeline.get_f0 (loads RMVPE lazily on first call)
    f0_coarse, f0_float = pipeline.get_f0(audio_f32, p_len, "rmvpe", 0)
    sync(dev)
    m["f0"] = (time.perf_counter() - t0) * 1000.0

    # glue — their pitch tensor conversion from Pipeline.pipeline
    t0 = time.perf_counter()
    pitch = torch.tensor(f0_coarse, device=dev).unsqueeze(0).long()
    pitchf = torch.tensor(f0_float, dtype=torch.float32, device=dev).unsqueeze(0)
    m["glue"] = (time.perf_counter() - t0) * 1000.0

    with torch.no_grad():
        # embed stage — their voice_conversion head
        t0 = time.perf_counter()
        feats = torch.from_numpy(audio_f32).float().view(1, -1).to(dev)
        feats = conv.hubert_model(feats)["last_hidden_state"]
        sync(dev)
        m["embed"] = (time.perf_counter() - t0) * 1000.0

        # faiss stage — their clone + retrieval + blend (h2d/d2h included)
        t0 = time.perf_counter()
        feats0 = feats.clone()
        feats = pipeline._retrieve_speaker_embeddings(
            feats, index, big_npy, index_rate
        )
        sync(dev)
        m["faiss"] = (time.perf_counter() - t0) * 1000.0

        # gen stage — their upsample + p_len + generator tail
        t0 = time.perf_counter()
        feats = F.interpolate(feats.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)
        feats0 = F.interpolate(feats0.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)
        p_len = min(audio_f32.shape[0] // WINDOW, feats.shape[1])
        p_len_t = torch.tensor([p_len], device=dev).long()
        audio1 = (
            (conv.net_g.infer(feats.float(), p_len_t,
                              pitch[:, :p_len], pitchf[:, :p_len].float(),
                              sid_t)[0][0, 0])
            .data.cpu().float().numpy()
        )
        sync(dev)
        m["gen"] = (time.perf_counter() - t0) * 1000.0

    return m


def summarize(xs, ws):
    rtf = [v / (ws * 1000.0) for v in xs]
    return {
        "mean_ms": statistics.mean(xs),
        "p50_ms": pctl(xs, 50),
        "p95_ms": pctl(xs, 95),
        "max_ms": max(xs),
        "rtf_mean": statistics.mean(rtf),
        "rtf_p95": pctl(rtf, 95),
    }


def cell(d):
    return f"{d['mean_ms']:7.1f} {d['p95_ms']:7.1f} {d['max_ms']:6.1f}"


def main():
    args = ARGS

    cfg = inference_config()
    if args.device:
        cfg.device = args.device
        if args.device != "mps":
            cfg.x_pad, cfg.x_query, cfg.x_center, cfg.x_max = (1, 6, 38, 41)
        else:
            cfg.x_pad, cfg.x_query, cfg.x_center, cfg.x_max = (1, 5, 30, 32)
    print(f"device: {cfg.device}   windows: {args.windows}   context pad: ±{args.pad} s"
          f"   reps: {args.reps}(+{args.warmup} warm-up)   index-rate: {args.index_rate}")
    print(f"model:  {args.model}")

    conv = VoiceConverter(config=cfg)
    t0 = time.perf_counter()
    conv.get_vc(args.model, 0)
    print(f"model load  (their get_vc -> load_model/setup_network): "
          f"{time.perf_counter() - t0:6.1f} s   tgt_sr={conv.tgt_sr}  version={conv.version}")

    pipeline = conv.vc

    global index, big_npy
    t0 = time.perf_counter()
    index = faiss.read_index(args.index)
    big_npy = index.reconstruct_n(0, index.ntotal)
    print(f"faiss index load: {time.perf_counter() - t0:6.1f} s   "
          f"{index.ntotal} vectors, dim={index.d}")

    t0 = time.perf_counter()
    conv.load_hubert("contentvec")
    print(f"contentvec load:  {time.perf_counter() - t0:6.1f} s")

    stream_src = load_audio_infer(args.audio, 16000)
    tile_n = math.ceil((16000.0 * 70) / len(stream_src))
    stream = np.tile(stream_src, tile_n)[: 16000 * 70]
    print(f"speech stream: {len(stream) / 16000:.0f} s @16 kHz "
          f"(from {os.path.basename(args.audio)}, {len(stream_src) / 16000:.1f} s, tiled)")

    sid_t = torch.tensor(0, device=cfg.device).unsqueeze(0).long()
    pad_n = int(args.pad * 16000)
    results = {}

    for ws in [float(x) for x in args.windows.split(",")]:
        win_n = int(ws * 16000)
        chunks = []
        off = 0
        while len(chunks) < args.reps + args.warmup and off + win_n + 2 * pad_n < len(stream):
            chunks.append(np.pad(stream[off : off + win_n], (pad_n, pad_n), mode="reflect"))
            off += win_n

        fail = None
        try:
            for c in chunks[: args.warmup]:  # warm-up: RMVPE lazy-load, MPS kernels, faiss
                run_chunk(pipeline, conv, c, sid_t, args.index_rate)

            per_stage = {s: [] for s in STAGES}
            totals = []
            for c in chunks[args.warmup :]:
                m = run_chunk(pipeline, conv, c, sid_t, args.index_rate)
                for s in STAGES:
                    per_stage[s].append(m[s])
                totals.append(sum(m[s] for s in STAGES))
            res = {s: summarize(per_stage[s], ws) for s in STAGES}
            res["total"] = summarize(totals, ws)
        except (RuntimeError, NotImplementedError) as e:
            print(f"  [{ws:g} s window FAILED: {e}]")
            fail = str(e)
            res = {"failed": fail}
        results[f"{ws:g}"] = res

    print(f"\n=== per-chunk inference, {cfg.device}, context pad ±{args.pad} s"
          f"  (n={args.reps}) ===")
    print(f"{'win_s':>6} | {'f0 (RMVPE)':>24} | {'embed (ContentVec)':>24} | "
          f"{'faiss retrieval':>24} | {'gen (net_g.infer)':>24} | {'glue':>17} | "
          f"{'total e2e compute':>24} | RTF p95")
    for ws_str, r in results.items():
        if r.get("failed"):
            print(f"{float(ws_str):g} s".rjust(6), "| FAILED:", r["failed"][:80])
            continue
        print(f"{float(ws_str):g} s".rjust(6),
              "|", cell(r["f0"]),
              "|", cell(r["embed"]),
              "|", cell(r["faiss"]),
              "|", cell(r["gen"]),
              f"| {r['glue']['mean_ms']:5.1f} {r['glue']['p95_ms']:11.1f}".rjust(17),
              "|", cell(r["total"]),
              f"| {r['total']['rtf_p95']:.2f}x")
    print("columns per stage: mean_ms / p95_ms / max_ms over timed chunks; "
          "RTF = compute ms ÷ window ms (p95).")

    if args.json:
        # Record paths relative to the repo so results stay machine-independent.
        def rel(v):
            return os.path.relpath(v, REPO) if isinstance(v, str) and os.path.isabs(v) and v.startswith(REPO) else v
        recorded = {k: rel(v) for k, v in vars(args).items()}
        with open(args.json, "w") as f:
            json.dump({
                "args": recorded,
                "device": cfg.device,
                "results": results,
            }, f, indent=2)
        print(f"\nfull results -> {args.json}")


if __name__ == "__main__":
    main()
