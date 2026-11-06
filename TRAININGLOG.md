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
**2026-11-05 — Chunked real-time inference latency on MPS (SPEC §4.1 C3 / §4.2)**

- Harness: `scripts/bench-rvc.py` — times Applio's own inference internals
  (`rvc/infer/pipeline.py`: `get_f0`/RMVPE, ContentVec forward, faiss
  `_retrieve_speaker_embeddings`, `net_g.infer`), stage by stage, per chunk;
  only the meters are local (C7 discipline). n=25 timed chunks per window
  size after 5 discarded warm-up chunks; larger window-size JSONs under
  `scripts/results/`.
- Machine: Apple M3 Pro (12 cores, 36 GB), macOS 14.6.1, torch MPS FP32;
  same box as Run 1. Input: tiled speech from
  `data/gvs/selected/yfdtmEbu8OA_3-.59-.38_4-.01-.30.wav` (the known-good
  A/B diagnostic file), trimmed to window + context; index-rate 0.75
  (whole-file whole-pipeline settings).

| window | context | f0 (RMVPE) | embed | faiss | generator | **total p95** | RTF p95 |
|---|---|---|---|---|---|---|---|
| 0.1 s | ±0.25 s | 22 ms | 8 ms | 3 ms | 37 ms | **69 ms** | 0.69× |
| 0.25 s | ±0.25 s | 27 ms | 9 ms | 3 ms | 43 ms | **81 ms** | 0.32× |
| 0.5 s | ±0.25 s | 29 ms | 10 ms | 4 ms | 51 ms | **92 ms** | 0.18× |
| 1.0 s | ±0.25 s | 33 ms | 13 ms | 5 ms | 70 ms | **120 ms** | 0.12× |

(columns: mean / p95 / max over timed chunks; extra ±0.5 s-context sweep
saved in `scripts/results/bench-pad05.json`, consistent +20–30 ms.)

**Findings**
1. Sustained chunk throughput is real-time on MPS with room: RTF 0.12–0.69.
   The older 0.6–0.8× whole-file figure included model load (≈10 s once) and
   a whole-file f0 pass — chunk-scale numbers after warm-up are much better.
2. **MPS hard limit discovered:** generator (HiFi-GAN decoder) output length
   = conv1d output channels; MPS supports ≤ 65536 channels → maximum ~1.64 s
   of audio per generator call at 40 kHz. Consequence: Applio's whole-file
   context padding (t_pad = ±1.0 s per side) **cannot run chunked on MPS at
   all** (fails from 0.1 s chunks up; ±0.5 s context additionally kills the
   1.0 s window). The chunker (C3) must keep window + 2·context ≤ ~1.5 s.
3. End-to-end latency is dominated by chunk accumulation: a chunked core
  can't start before the whole window is captured, so e2e ≈ window + compute +
   I/O buffers (≈ 30 + 25 ms) ≈ 220 ms @0.1 s, ≈ 380 ms @0.25 s, ≈ 470 ms
   @0.5 s. The 150–200 ms target ⇒ 0.1 s windows is the only knob in range
   on this hardware; 0.25 s sits just over the 300 ms ceiling.
4. Stage anatomy: RMVPE and generator scale with window+context; faiss and
   ContentVec nearly flat. No CUDA-only blockers hit on the
   `rvc.infer` path; Applio's `rvc/realtime` path (CUDAGraph/fp16) remains
   off-limits for MPS but wasn't needed for the measurement.
5. Not yet measured: with a live audio loop (C1/C4 buffers), true wall
   latency; 10 s–window quality (the next audition); CPU device comparison.

**Next:** audition 0.1 s-window chunks (single-variable vs Run 1 whole-file
rendition) to lock the Fast preset window size, then write the C3 chunker
design doc (window/context/crossfade + the ≤65536-sample decoder limit).
