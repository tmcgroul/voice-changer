# Training log

One entry per fine-tune run, in a fixed template, so runs can be compared
side by side: what changed between them and what each change did. Append a
new entry per run — do not rewrite old ones.

- Verdict hierarchy: ears > loss numbers > everything else. Loss is
  convergence evidence, not quality evidence (see the one-off notes below).
- Machine-specific numbers (epoch time, download size) are context, not
  requirements.

## Template

```
### Run <N>: <voice> — <YYYY-MM-DD>
Base model:       (pretrained pair + rate + vocoder, checkpoint source)
Tool:             (Applio commit/ecc repo state, python, pip-only notes)
Dataset:          (source, #clips, total minutes, sr/channels, demucs?)
Train settings:   (epochs, batch, f0 method, device, index algorithm)
Timing:           (download / preprocess / extract / train per-epoch+total)
Loss track:       (start → best@epoch, shape/plateau)
Checkpoints kept: (paths)
Output model:     (final pth + index path)
Audition:         (what was listened to, verdict per sample, defects)
Next-tune ideas:  (what to change next run and why)
```

---

### Run 1: gvs — 2026-11-05

**Base model:** RVC v2 pretrained pair `f0G40k.pth` + `f0D40k.pth` (40 kHz,
HiFi-GAN vocoder), from Applio's HuggingFace `IAHispano/Applio` →
`Resources/pretrained_v2/` (RVC-Project's official v2 speech base). The
`core.py prerequisites` call downloaded all rates' pairs (1.91 GB total);
only the 40k pair was used.

**Tool:** Applio clone (main @ `6d5d9511`, 2026-11-04), invendored at
`external/Applio`. Python 3.12.2 venv via uv; requirements pinned
(torch 2.11.0). Fully pip/uv — no Homebrew (the `faiss-cpu` pip wheel
replaces the install script's `brew install faiss` step).

**Dataset:** `data/gvs/selected/` — 9 clips from a single YouTube source
(`yfdtmEbu8OA`, cut via `scripts/extract-audio.sh` + cutlist), **9.1 min**
total, 44.1 kHz stereo PCM → resampled to 40 kHz mono during preprocess.
Deliberately under the 20-min goal in SPEC §6 — getting a baseline was the
priority; growing the cutlist runs in parallel. No `--demucs` needed
(clean-speech selections only).

**Train settings:** 100 epochs, batch 8, RMVPE f0, Faiss index algorithm,
preprocess chunk 3.0 s / overlap 0.3 s, `--save-only-latest` +
`--save-every-epoch 25`. Device: Apple-silicon MPS (FP32, spectral loss
computed on CPU). Index built at inference rate 0.5 (test setting, see
Audition below).

**Timing:** dep install ≈ 5 min once; 1.9 GB model download 53 s;
preprocess 12 s; f0+embedder extract ≈ 25 s; training ≈ 42 s/epoch →
~70 min for 100 epochs. (First attempt was session-killed at ep 42;
auto-resume from checkpoint after relaunch, exactly as documented.)

**Loss track:** 22.35 (ep 26) → 21.10 (ep 28) → best **18.66 @ ep 56**, flat
afterwards. Adversarial training, so this is convergence evidence only; the
curve flattening was the visible "you've got enough epochs" signal.

**Checkpoints kept:** `data/gvs/models/gvs/` — `gvs_25e_675s.pth`,
`gvs_50e_1350s.pth`, `gvs_75e_2025s.pth`, `gvs_100e_2700s.pth`, plus
`gvs.index` (faiss over training chunks), intermediate G/D latest snapshots.

**Output model:** `data/gvs/models/gvs/gvs_100e_2700s.pth` + `gvs.index`
(reconstruction `rt-`* renditions were made from the 100e model; the 25e
one (`rt25-gvs.wav`) for epoch-count comparison came from `gvs_25e_675s.pth`).

**Inference (whole-file CLI pass, per-run reference, no chunking):**
~0.6–0.8× realtime on MPS for `--pitch 0 --index-rate 0.5 --f0-method rmvpe`
(165 s clip → 127 s; includes model+index load overhead — not a latency
verdict; real-time chunked measurement is a later §4.2 concern).

**Audition:** three input-reconstruction renditions (train clips fed back
through the model, `/tmp/rt-*.wav` next to the originals in
`data/gvs/selected/`) plus a 25-epoch variant of one clip (`rt25-gvs.wav`).
Owner verdict: **"sound ok but needs to be improved"**; follow-up: delivery
and tone are right (not robotic), the defect is **extra "noise" vs the
original** (additive artifacts layer). Diagnostic A/B set (sample
`yfdtmEbu8OA_4-.08-.16_4-.09-.30.wav`): `data/gvs/audition/rtA_ix0.wav` = same model,
index-rate 0; `data/gvs/audition/rtB_75e.wav` = 75-epoch checkpoint, index-rate 0.5.
Listening result: **no audible difference across all three** — index-blend
artifacts and late-epoch overfit are both ruled out; remaining suspects are
the generator/base-model noise floor and the source recordings themselves
(possible quiet clicks/fan noise in the YouTube source). Owner decision:
accepted for now — no further tuning of Run 1.

**Next-tune ideas (updated after A/B verdict "no audible difference"):**
- Inference knobs (index-rate) and epoch count are *not* the noise levers.
  Run 1 stays as-is (owner: happy with current quality for now).
- If the extra noise needs fixing later, candidates are dataset-side:
  grow the cutlist toward 20+ min; optional source cleanup for click/fan
  noise (ffmpeg `adeclick`/`afftdn` in extract-audio.sh, designed but not
  implemented); then retrain and compare in a new entry.
- Priority shifts to SPEC §7's other leg: measure chunked real-time
  inference latency on MPS against the §4.2 budget.

---

## One-off notes (non-run records, useful background)

- "Loss" in the log = the training-time error score the optimizer
  minimizes (spectral + adversarial + feature-matching blended). Lower is
  better *within a run*; absolute values are not comparable across runs,
  datasets, or rates.
- f0/rate/vocoder mismatch is real: all timing/steps above were measured
  with `--sample-rate 40000` end-to-end; trying different recipes scales
  roughly linearly in epochs so long as the dataset size is the baseline
  variable (don't change two variables between auditions).
- Machine at Run 1: Apple M3 Pro, macOS 14.6.1, 12 cores, 36 GB unified
  memory, ~0.6–0.8× realtime whole-file MPS inference as noted.

## Benchmark notes (not a training run; uses the Run 1 model)
**2026-11-05 — Chunked real-time inference latency on MPS (SPEC §4.2.1)**

Method: `scripts/bench-rvc.py` times Applio's own inference internals stage
by stage (RMVPE f0, ContentVec, faiss retrieval, `net_g.infer`; only the
meters are local — C7 discipline). n=25 timed chunks/window after 5 warm-up
chunks. Machine: M3 Pro (12 cores, 36 GB), macOS 14.6.1, MPS FP32, same box
as Run 1. Input: tiled speech from `yfdtmEbu8OA_3-.59-.38_4-.01-.30.wav`,
index-rate 0.75, context ±0.25 s. Raw JSON: `data/gvs/benchmarks/`.

| window | f0 (RMVPE) | embed | faiss | generator | **total p95** | RTF p95 |
|---|---|---|---|---|---|---|
| 0.1 s | 22 ms | 8 ms | 3 ms | 37 ms | **69 ms** | 0.69× |
| 0.25 s | 27 ms | 9 ms | 3 ms | 43 ms | **81 ms** | 0.32× |
| 0.5 s | 29 ms | 10 ms | 4 ms | 51 ms | **92 ms** | 0.18× |
| 1.0 s | 33 ms | 13 ms | 5 ms | 70 ms | **120 ms** | 0.12× |

(columns: mean/p95/max over timed chunks; a ±0.5 s-context sweep in
`bench-pad05.json` shows +20–30 ms.)

**Findings**
1. Real-time on MPS with room (RTF 0.12–0.69); the older 0.6–0.8× whole-file
   figure included model load + full-file f0 and no longer applies.
2. **MPS hard limit:** generator output = conv1d channels ≤ 65536 → max
   ~1.64 s of audio per call @40 kHz; Applio's ±1.0 s padding fails on MPS
   at every chunk size (±0.5 s additionally kills 1.0 s windows). Chunker
   (C3) must keep window + 2·context ≤ ~1.5 s.
3. e2e ≈ window + compute + I/O ≈ 220 ms @0.1 s, ~380 ms @0.25 s → the
   150–200 ms target requires 0.1 s windows on this hardware.
4. No CUDA-only blockers on the `rvc.infer` path (`rvc/realtime` remains
   CUDA-only, untouched). Not yet measured: live-loop wall latency,
   short-window quality (next audition), CPU device comparison.

**Next:** audition 0.1 s-window chunks (single-variable vs Run 1 whole-file
rendition) to lock the Fast preset window size, then write the C3 chunker
design doc (window/context/crossfade + the ≤65536-sample decoder limit).

## Audition (pending owner verdict)
**2026-11-07 — window size: 0.1 s chunks vs whole file (Fast-preset quality gate)**

Single variable = chunking scheme (§4.2.1's open question). Same clip, model
`gvs_100e_2700s.pth`, rmvpe, index-rate 0.75, protect 0.5; **both rendered on
CPU** (Applio's whole-file path can't run on MPS at >1.64 s output — see §4.2.1
finding 2 — and device must not be the A/B variable).

- Source: `~/ma-voice.m4a` (6.66 s, owner's fresh recording) → clean copy
  `data/gvs/audition/ma-voice.wav`.
- A: `data/gvs/audition/rendA-whole.wav` — one pass, Applio whole-file
  pipeline (`scripts/render-audition.py --mode whole`).
- B: `data/gvs/audition/rendB-win010.wav` — 0.1 s windows, hop = window,
  context ±0.25 s stitched back (`--mode chunked --window 0.1`).
- Automated seam check: RMS diff at 0.1 s boundaries vs within-chunk frames —
  same magnitude (p95 3.23 vs 3.39 rel-diff), zero-crossing profiles match →
  no obvious clicks; repeatable numbers live in the script.

Owner verdict: **pending**.
- If B ≈ A → lock Fast preset = 0.1 s windows; next: C3 chunker design doc.
- If B audibly worse → 0.1 s out; candidates: 0.25 s windows (est. ~380 ms e2e,
  over ceiling), compute trim (FCPE f0, slimmer context), or accept.

**Owner verdict (listened 2026-11-07): B (0.1 s windows) ILLEGIBLE — rejected.**
Fast preset at 0.1 s windows is out. Next listening round: 0.25 s and 0.5 s
windows (same renderer, single-variable again) to find the smallest acceptable
window before locking a preset.
