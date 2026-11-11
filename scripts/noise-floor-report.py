#!/usr/bin/env python3
"""Noise-floor diagnostic for the "no fan-like noise during silence" defect.

Renders test inputs (real pause with fan noise / digital silence /
pause->silence / speech) through the fine-tuned model and a wrapped base
pretrained generator, then reports per-file frame-RMS floor and octave-band
energy. Outputs live in data/gvs/audition/noisetest/; numbers printed below
match the TRAININGLOG.md diagnostic entry (2026-11-09).

Inputs are built by the same script from data/gvs/selected (idempotent).
Renders require scripts/render-audition.py (Applio venv); wrapper checkpoint
(external/Applio/logs/base-pretrained.pth) is rebuilt from the stock
hifi-gan pretrain if missing.

Run:
  external/Applio/.venv/bin/python scripts/noise-floor-report.py
"""
import math
import os
import struct
import subprocess
import sys
import wave

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APPLIO = os.path.join(REPO, "external", "Applio")
SRC_CLIP = os.path.join(REPO, "data", "gvs", "selected",
                        "yfdtmEbu8OA_3-.59-.38_4-.01-.30.wav")
NT = os.path.join(REPO, "data", "gvs", "audition", "noisetest")
GVS = os.path.join(REPO, "data", "gvs", "models", "gvs", "gvs_100e_2700s.pth")
GIDX = os.path.join(REPO, "data", "gvs", "models", "gvs", "gvs.index")
BASE_PTH = os.path.join(APPLIO, "logs", "base-pretrained.pth")
PAUSE = (27.85, 31.65)   # measured quiet span (fan noise present) in SRC_CLIP

RENDER = os.path.join(REPO, "scripts", "render-audition.py")

# ---------- build inputs -----------------------------------------------------
def read_mono(path):
    with wave.open(path) as w:
        n, ch, fs = w.getnframes(), w.getnchannels(), w.getframerate()
        d = struct.unpack("<%dh" % (n * ch), w.readframes(n))
    return [sum(d[i * ch:(i + 1) * ch]) // ch for i in range(n // ch)], fs


def write16(path, samples, fs):
    with wave.open(path, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(fs)
        w.writeframes(struct.pack("<%dh" % len(samples), *samples))


def frames_db(x, fs, win=0.05):
    w = int(fs * win)
    return [10 * math.log10(sum(v * v for v in x[i:i + w]) / w + 1e-24)
            for i in range(0, len(x) - w, w)]


os.makedirs(NT, exist_ok=True)
m, fs = read_mono(SRC_CLIP)
p0, p1 = int(PAUSE[0] * fs), int(PAUSE[1] * fs)
pause = m[p0:p1]
win = int(fs * 0.05)
rdb = frames_db(m, fs)
start = max(range(0, len(rdb) - int(4 / 0.05)),
            key=lambda i: sum(rdb[i:i + int(4 / 0.05)]))
speech = m[start * win:start * win + int(4 * fs)]
write16(f"{NT}/in_pause.wav", pause, fs)
write16(f"{NT}/in_pause_sil.wav", pause[:int(1.2 * fs)] + [0] * int(3.5 * fs), fs)
write16(f"{NT}/in_sil.wav", [0] * int(3.5 * fs), fs)
write16(f"{NT}/in_voice.wav", speech, fs)

# ---------- wrapper checkpoint for the stock base pretrain --------------------
if not os.path.exists(BASE_PTH):
    import torch
    g = torch.load(GVS, map_location="cpu", weights_only=True)
    b = torch.load(os.path.join(APPLIO, "rvc/models/pretraineds/hifi-gan/f0G40k.pth"),
                   map_location="cpu", weights_only=True)
    b = b["model"] if "model" in b else b
    cfg = list(g["config"]); cfg[-3] = b["emb_g.weight"].shape[0]  # n_spk
    torch.save({"weight": b, "config": cfg, "sr": 40000, "f0": 1,
                "version": "v2", "vocoder": "HiFi-GAN"}, BASE_PTH)
    print("built", BASE_PTH)

# ---------- render ------------------------------------------------------------
PY = os.path.join(APPLIO, ".venv", "bin", "python")
for t in ("pause", "pause_sil", "sil", "voice"):
    for tag, model, irate in (("gvs", GVS, 0.75), ("base", BASE_PTH, 0.0)):
        out = f"{NT}/o_{tag}_{t}.wav"
        if os.path.exists(out):
            continue
        subprocess.run([PY, RENDER, "--src", f"{NT}/in_{t}.wav", "--mode", "chunked",
                        "--window", "0.25", "--device", "mps",
                        "--model", model, "--index", GIDX, "--index-rate", str(irate),
                        "--out", out], cwd=APPLIO, check=True)

# ---------- report ------------------------------------------------------------
import numpy as np          # Applio venv
import soundfile as sf

BANDS = [(50, 100), (100, 200), (200, 400), (400, 800), (800, 1600),
         (1600, 3150), (3150, 6300), (6300, 12500)]
files = ["in_pause.wav", "in_sil.wav"] + \
        [f"o_{tag}_{t}.wav" for tag in ("gvs", "base")
         for t in ("pause", "pause_sil", "sil", "voice")]
print(f"{'file':22s} {'floor':>6s} {'med':>6s} | bands: " +
      " ".join(f"{lo}-{hi}" for lo, hi in BANDS))
for f in files:
    x, fsx = sf.read(os.path.join(NT, f))
    x = x if x.ndim == 1 else x[:, 0]
    w = int(fsx * 0.05); n = (len(x) - w) // w
    fr = np.stack([x[i * w:(i + 1) * w] for i in range(n)])
    rms = 10 * np.log10((fr ** 2).mean(1) + 1e-24)
    X = np.mean([np.abs(np.fft.rfft(fr[i] * np.hanning(w))) ** 2 for i in range(n)], 0)
    kspec = np.arange(w // 2 + 1) * fsx / w
    bands = [10 * np.log10(((kspec >= lo) & (kspec < hi)).dot(X)
                           / max(1, ((kspec >= lo) & (kspec < hi)).sum()) + 1e-24)
             for lo, hi in BANDS]
    p10, p50 = np.percentile(rms, 10), np.percentile(rms, 50)
    print(f"{f:22s} {p10:6.1f} {p50:6.1f} | " + " ".join(f"{v:6.0f}" for v in bands))
