#!/usr/bin/env python3
"""Render audition A/B renditions of one source clip (SPEC §4.2.1 open item).

Two modes, identical model/settings, only the chunking scheme differs:

  whole   — one pass through Applio's Pipeline.pipeline (Run-1 audition style)
  chunked — the clip sliced into --window s windows, hop = window, context
            --context s on each side, middle-of-window output stitched back.

The listener compares the two renditions to answer: does 0.1 s windowing hurt
quality? (bench-rvc.py showed it is fast enough; quality was unmeasured.)

Settings match the bench: the fine-tuned model under test, rmvpe f0,
index-rate 0.75, protect 0.5, contentvec, MPS by default.

Example:
  external/Applio/.venv/bin/python scripts/render-audition.py \
      --src data/<voice>/audition/ma-voice.wav --mode whole \
      --out data/<voice>/audition/rendA-whole.wav \
      --model data/<voice>/<voice>_100e_2700s.pth --index data/<voice>/<voice>.index
  external/Applio/.venv/bin/python scripts/render-audition.py \
      --src data/<voice>/audition/ma-voice.wav --mode chunked --window 0.1 \
      --out data/<voice>/audition/rendB-win010.wav \
      --model data/<voice>/<voice>_100e_2700s.pth --index data/<voice>/<voice>.index
"""

import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APPLIO = os.path.join(REPO, "external", "Applio")

import argparse

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--src", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--mode", choices=("whole", "chunked"), default="whole")
ap.add_argument("--window", type=float, default=0.1, help="chunked mode: window seconds (hop = window)")
ap.add_argument("--context", type=float, default=0.25, help="chunked mode: context seconds per side")
ap.add_argument("--model", required=True, help="fine-tuned .pth to render with")
ap.add_argument("--index", required=True, help="matching faiss .index file")
ap.add_argument("--index-rate", type=float, default=0.75)
ap.add_argument("--pitch", type=int, default=0)
ap.add_argument("--device", choices=("mps", "cpu"), default=None)
ap.add_argument("--protect", type=float, default=0.5,
                help="Protect voiceless consonants (0..0.5; UI knob)")
ap.add_argument("--rms-mix-rate", type=float, default=1.0,
                help="Volume envelope blend (0..1; UI knob)")
ap.add_argument("--f0-method", default="fcpe", choices=("rmvpe", "fcpe", "crepe"),
                help="Run-2 owner ear verdict: fcpe preferred (2026-11-10)")
ap.add_argument("--cleanup-mode", default="off", choices=("off", "light", "speech"),
                help="Input-stage audio cleanup (same as UI Audio cleanup)")
ap.add_argument("--cleanup-strength", type=float, default=12)
ARGS = ap.parse_args()
# Resolve relative paths before Applio changes cwd out from under us.
ARGS.src = os.path.abspath(ARGS.src)
ARGS.out = os.path.abspath(ARGS.out)
ARGS.model = os.path.abspath(ARGS.model)
ARGS.index = os.path.abspath(ARGS.index)

# Applio resolves embedder/predictor assets relative to cwd.
os.chdir(APPLIO)
sys.path.insert(0, APPLIO)

import faiss  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import soundfile as sf  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from rvc.configs.config import inference_config  # noqa: E402
from rvc.infer.infer import VoiceConverter  # noqa: E402
from rvc.lib.utils import load_audio_infer  # noqa: E402

cfg = inference_config()
if ARGS.device:
    cfg.device = ARGS.device
    if ARGS.device != "mps":
        cfg.x_pad, cfg.x_query, cfg.x_center, cfg.x_max = (1, 6, 38, 41)
    else:
        cfg.x_pad, cfg.x_query, cfg.x_center, cfg.x_max = (1, 5, 30, 32)

conv = VoiceConverter(config=cfg)
conv.get_vc(ARGS.model, 0)
index = faiss.read_index(ARGS.index)
big_npy = index.reconstruct_n(0, index.ntotal)
conv.load_hubert("contentvec")
pipeline = conv.vc
assert conv.tgt_sr == 40000, conv.tgt_sr

src = load_audio_infer(ARGS.src, 16000)
n_in = len(src)
print(f"device={cfg.device} mode={ARGS.mode} src={n_in/16000:.2f} s")


def save(audio, note=""):
    out = os.path.abspath(ARGS.out)
    sf.write(out, audio, conv.tgt_sr)
    print(f"saved {out}  ({len(audio)/conv.tgt_sr:.2f} s @ {conv.tgt_sr} kHz) {note}")


if ARGS.mode == "whole":
    # Optional input-stage cleanup (same helper the UI cleanup knob uses).
    if ARGS.cleanup_mode != "off":
        from rvc.lib.tools.audio_cleanup import cleanup_audio  # noqa: E402

        src = cleanup_audio(src, 16000, ARGS.cleanup_mode, ARGS.cleanup_strength)
        print(f"input cleanup: {ARGS.cleanup_mode} @{ARGS.cleanup_strength} dB")
    # Their whole-file path, verbatim (their defaults: peak-norm 0.99 etc.)
    audio_max = np.abs(src).max() / 0.95
    if audio_max > 1:
        src = src / audio_max
    audio_opt = pipeline.pipeline(
        model=conv.hubert_model,
        net_g=conv.net_g,
        sid=0,
        audio=src,
        pitch=ARGS.pitch,
        f0_method=ARGS.f0_method,
        file_index=ARGS.index,
        index_rate=ARGS.index_rate,
        pitch_guidance=conv.use_f0,
        volume_envelope=ARGS.rms_mix_rate,
        version=conv.version,
        protect=ARGS.protect,
        f0_autotune=False,
        f0_autotune_strength=1.0,
        proposed_pitch=False,
        proposed_pitch_threshold=155.0,
    )
    save(audio_opt, "(their whole-file pipeline)")
    sys.exit(0)

# ---- chunked mode: same ops bench-rvc.py times, one voice_conversion/chunk ----
win = int(ARGS.window * 16000)            # input window @16 kHz
ctx = int(ARGS.context * 16000)           # context per side
ratio = conv.tgt_sr / 16000.0             # input->output sample ratio (2.5 @40k)
dev = cfg.device
sid_t = torch.tensor(0, device=dev).unsqueeze(0).long()


def keep_len(n_win_samples):
    return int(round(n_win_samples * ratio))


pieces = []
off = 0
i = 0
while off < n_in:
    n_win = min(win, n_in - off)
    chunk = np.pad(src[off : off + n_win], (ctx, ctx), mode="reflect")
    p_len_pad = len(chunk) // 160
    with torch.no_grad():
        f0_coarse, f0_float = pipeline.get_f0(chunk.astype(np.float32), p_len_pad, "rmvpe", ARGS.pitch)
        pitch = torch.tensor(f0_coarse, device=dev).unsqueeze(0).long()
        pitchf = torch.tensor(f0_float, dtype=torch.float32, device=dev).unsqueeze(0)
        feats = torch.from_numpy(chunk.astype(np.float32)).float().view(1, -1).to(dev)
        feats = conv.hubert_model(feats)["last_hidden_state"]
        feats0 = feats.clone()
        feats = pipeline._retrieve_speaker_embeddings(feats, index, big_npy, ARGS.index_rate)
        feats = F.interpolate(feats.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)
        feats0 = F.interpolate(feats0.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)
        p_len = min(len(chunk) // 160, feats.shape[1])
        p_len_t = torch.tensor([p_len], device=dev).long()
        out_i = (
            (conv.net_g.infer(feats.float(), p_len_t,
                              pitch[:, :p_len], pitchf[:, :p_len].float(),
                              sid_t)[0][0, 0])
            .data.cpu().float().numpy()
        )
    if dev == "mps":
        torch.mps.synchronize()
    # trim the synthesized context, keep exactly this window's span
    trim = ctx * ratio
    keep = keep_len(n_win)
    piece = out_i[int(round(trim)) : int(round(trim)) + keep]
    if len(piece) < keep:  # tail safety
        piece = np.pad(piece, (0, keep - len(piece)))
    pieces.append(piece)
    if (i := i + 1) % 10 == 0:
        print(f"  chunk {i}: {off+n_win:.0f}/{n_in} in-samples")
    off += n_win

audio_opt = np.concatenate(pieces)
audio_max = np.abs(audio_opt).max() / 0.99
if audio_max > 1:
    audio_opt /= audio_max
save(audio_opt, f"(chunked: window {ARGS.window} s, hop=window, context ±{ARGS.context} s)")
