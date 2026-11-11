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

### Run 2: gvs (trimmed + silence anchor) — 2026-11-09

**Environment:** same as Run 1 (Applio main, 40k v2 pretrain, MPS); venv
rebuilt 2026-11-09 after the venv was found stale (old `third_party/` path
from before the repo move — recreated, requirements reinstalled,
prerequisites ~2 GB still cached).

**What Run 2 changes vs Run 1 (two dataset-composition changes, one
"silence policy" theme):** (1) quiet-gap trimming applied to every cut
(−60 dB / ≥0.25 s / keep 0.08 s — owner-approved by listening to discarded
streams, see diagnostic above); dataset 9.4 → 6.9 min pre-slice;
(2) `--include-mutes 10` (30 s silence anchor vs the default 6 s).
Voice-affecting DSP: none (no denoiser — owner rejected all variants).
Pre-trim copies of all 9 clips kept at `data/gvs/selected-run1/`;
Run-1 checkpoints + index kept at `data/gvs/models/gvs-run1/`.

**Dataset:** same 9 clips, `data/gvs/selected/` (trimmed).
**Train settings:** 100 epochs, batch 8, rmvpe, MPS, faiss (unchanged).
**MPS patch:** first attempt CRASHED at the epoch-24 evaluation render —
`NotImplementedError: Output channels > 65536 not supported at the MPS
device` inside the tensorboard reference render (`net_g.infer(*reference)`,
triggers only when `epoch % save_every_epoch == 0`); Run 1 got through only
because its random reference segments were short. Patch applied in
`external/Applio/rvc/train/train.py`: the eval render try/excepts the MPS
error and falls back to a one-off CPU deepcopy render (training loop
untouched; same patch survives epoch 25 in the restarted run). ~24 epochs
of the crashed attempt were lost (no checkpoint existed); restarted fresh.
**Timing:** extract ≈ 1 min; train ≈ 44 s/epoch (same as Run-1 pace),
~74 min for 100 epochs; ~24 epochs lost to the first attempt's eval crash.
**Loss track:** 16.70 (ep 14, first attempt — restarted) → 14.14 (ep 42)
→ 12.44 best @ ep 79, flat to ep 100. Loss numbers are not comparable
across runs/datasets (see one-off notes); shape-wise the plateau lands
earlier (ep ~42–79) than Run 1's ep 56, consistent with the smaller,
trimmed dataset.
**Checkpoints kept:** `data/gvs/models/gvs/` — `gvs_25e_700s.pth`,
`gvs_50e_1400s.pth`, `gvs_75e_2100s.pth`, `gvs_100e_2800s.pth` +
`gvs.index` (fresh — no Run-1 resume; Run-1 training state archived at
`data/gvs/models/gvs-run1/`).
**Output model:** `data/gvs/models/gvs/gvs_100e_2800s.pth` + `gvs.index`
(copies served to `external/Applio/logs/gvs/`).
**Audition (closed 2026-11-10):** rendered (same renderer/settings as
Run 1: chunked 0.25 s, MPS, rmvpe, index-rate 0.75): (a) pause/voice test
pair through r1 and r2 models (`data/gvs/audition/noisetest/
o_r{1,2}_{pause,voice}.wav`); (b) full-clip reconstructions
`data/gvs/audition/o_r{1,2}-rtA_3-.59.wav`; single-file loudness-matched
A/Bs: `noisetest/AB1-pause-r1-then-r2.wav`, `AB2-voice-r1-then-r2.wav`.
**Owner verdicts:** Convert-page overall: "almost identical to the old
one" — no audible noise improvement, no regression. Pause A/B file:
"sounds like shit, can't even understand the resulted audio" — note the
pause file is a converted *pause* (no speech input, so the output is the
noise bed itself, by design — it is a diagnostic, not a usability sample);
owner could not hear an improvement in it. **Run-2 conclusion: the
silence-policy changes (trim + 30 s mutes) did not audibly reduce the
noise bed**; dataset growth (Round 3) is the remaining lever for it.
**Parameter matrix verdict:** fcpe + protect + rms variants "best results"
(see follow-up for exact values baked into defaults).
**Parameter matrix (owner ear verdict pending):** render-audition.py
extended with the Convert-page knobs (--protect/--rms-mix-rate/--f0-method/
--cleanup-mode on CPU whole-file); 9 variants of a 12-s reference segment
rendered to `data/gvs/audition/params/v-*.wav` with listening guide
`LISTEN.md` (protect 0.35/0.25, rms 0.75/0.50, fcpe, crepe, light cleanup,
DFN3 input cleanup). Cleanup modes act on INPUT only — cannot fix the
model-generated bed (noted in guide).
**Next-tune ideas:** (1) Round 3 = cutlist growth: 12 draft 3-min blocks
(≈36 min, speech-dense zone 4:05–5:05, same recording chain) drafted at
`data/gvs/cutlist-round3-proposal.txt` — owner curates by ear, then extract
+ retrain (silence policy stays). (2) Structural fallbacks if Rounds 2–3
plateau: `spin-v2` embedder fine-tune (different speaker-feature extractor;
off-distribution risk) or the unused `refinegan` pretrained pair as
vocoder base (untested quality path). (3) Click pass (`adeclick`) as its
own one-variable round only if clicks survive the trimming.

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

**Owner verdict round 2 (2026-11-07):** `rendC-win025` still "sucks";
`rendD-win05` intelligible (actual words audible). => with this renderer
(no crossfade, edge-to-edge stitching), the window floor is ~0.5 s.

Open question before raising the floor: how much of the short-window
damage is stitching vs f0/embedding quality? Next single-variable test:
crossfade/overlap-add at a fixed window (0.25 s) vs rendC (same window,
no crossfade).
(Note per owner: while tweaking audition parameters, don't commit every
change; batch meaningful commits only.)

**Live smoke test via Applio's own realtime tab (not a script round)**
Same session. Model served from `external/Applio/logs/gvs/` (copies of the
Run 1 pth + index). Streaming worker ran on **MPS** (`Realtime computation
device: mps (FP32)`, `Realtime is ready!`, no per-block errors; non-fatal
torchfcpe debug noise in the log). The tab's record-test path ran into the
same MPS 65536-sample wall on its whole-file component and **auto-fell back
to CPU** (22.4 s for the recorded clip) — the Tw6pre-bench finding applied by
their own fallback logic (`is_mps_error`), pattern confirmed in the wild.

Owner verdicts:
- Quality live: **overall OK, slightly robotic timbre** — parked as fine-tune
  quality work (ties into the Run 1 "extra noise" defect family; the plan is
  improving content of fine-tuning: cutlist growth / cleanup, not architecture).
- Latency: default chunk 250 ms = "noticeable pause"; chunk 100 ms reduced
  delay with **equal or near-equal quality**. Remaining knobs identified but
  not tuned by ear: crossfade overlap (0.05 s), extra conversion context
  (2.5 s/block), playback buffer. Objective loopback latency measurement is
  the prerequisite for locking presets (deferred).

**Session conclusions (2026-11-07, real-time milestone)**
1. Real-time RVC conversion is **usable on this hardware today** via Applio's
   existing realtime engine — no need to build a C3 chunker from scratch.
   Decision: C3 = wrap/adopt Applio realtime engine (worker, SOLA +
   phase-vocoder crossfade, VAD, hot-swap), the same "reuse the tooling"
   stance C7 already takes.
2. Speed is settled (§4.2.1 bench + live confirmation); **quality is the
   frontier**: robotic tone/fine-tune stage is the next improvement target.
3. The edge-stitched audition grid (0.1/0.25/0.5 s without blending) stays
   valuable as a controlled diagnostic renderer, but its harsh verdicts are
   partly stitching, not window-size alone — the live engine with proper
   crossfade sounded acceptable where the naive renderer did not.
4. Deferred: scripted loopback latency measurement, preset locking, C5
   virtual mic.

Applio relaunch: `cd external/Applio && ./run-applio.sh --client` → Realtime tab.

## Diagnostic — where does the "fan noise" in live output come from?
**2026-11-09 — pause/noise-floor A/B (no retraining; dataset-side question)**

Trigger: owner's live realtime session — fan-like noise audible in output even
while mic input was silent. Follows Run 1's unresolved "extra additive noise"
hyper-norm (audition A/B had ruled out index-rate and epochs).

Method: `scripts/noise-floor-report.py` builds 4 test inputs from a real 3.8 s
pause (fan noise present) / digital silence / pause→silence / speech of clip
`yfdtmEbu8OA_3-.59-.38_4-.01-.30.wav`, renders each through (a) gvs_100e_2700s
index-rate 0.75, (b) a wrapped stock base pretrain (`rvc/models/pretraineds/
hifi-gan/f0G40k.pth` + metadata from the gvs checkpoint → `logs/base-
pretrained.pth`, index-rate 0) — chunked 0.25 s windows, MPS. All rendition
files in `data/gvs/audition/noisetest/` (in_* = inputs, o_* = outputs);
spectrogram PNGs alongside; numbers reproducible via the script.

| file | floor dBFS | character |
|---|---|---|
| in_pause (source) | −70 | fan hash, 200–800 Hz peak |
| o_gvs_pause | −84/−64 med | **same noise** at matched level/spectrum |
| o_gvs_sil (input = zeros) | −76 | model **still emits** a low-mid hash bed, fan-shaped band profile |
| o_gvs_pause_sil (pause then zeros) | −75 after 1 s | floor drops to a steady −75 bed, stays constant |
| o_base_sil | −33 | base pretrain emits a much louder low-rumble bed (50–200 Hz) |
| o_base_pause | −33/−25 med | base invents loud ambient on noise-only input |

Verdict (pending owner listening pass over noisetest/):
1. **The fine-tuned model synthesizes the floor during true silence** — it
   cannot be input leakage. Its spectral shape matches the source fan band
   (200–800 Hz emphasis), so the fine-tune learned the dataset's constant
   broadband floor as the "quiet state of this voice". This closes the
   dataset-noise suspect: it is in the model now; only dataset-side cleanup +
   retrain removes it (inference knobs confirmed irrelevant, again).
2. But a silence bed is **partly architectural**: the stock base pretrain
   emits an even louder bed (−33 dBFS, rumble-weighted). Cleanup should make
   the bed quieter/more neutral, not zero.
3. On real pauses the gvs model reproduces the source fan noise pass-through
   style — so pauses in converted speech keep their exact room signature.

Consequences: implement planned source cleanup (`adeclick`/`afftdn` chain in
extract-audio.sh, high-pass first), retrain, A/B renditions' pause floors;
short-term live usability lever = output noise gate (masks bed between
phrases, does nothing for noise riding on speech).

**Follow-up A (2026-11-09 later same day): owner listened and rejected ALL
denoising** — auditioned on 20-s snippet of `yfdtmEbu8OA_4-.08-.16_4-
.09-.30.wav`: high-pass-only ≈ no difference; afftdn10 = underwater;
RNNoise std = spontaneous artifacts + robotic voice; RNNoise mp = weird
intra-word pauses; RNNoise+afftdn6 = voice changed. No denoiser variant
acceptable. Conclusion: any per-sample DSP over speech is off the table
for this use case; attack noise only via *dataset composition*.

**Follow-up B (same day): quiet-gap trimming — auditioned and approved.**
Mechanism: instead of denoising (rejected in Follow-up A), *remove quietly
noisy time spans from the dataset*: ffmpeg `silenceremove` deletes stretches
whose RMS stays below a threshold for ≥0.25 s (keeping 0.08 s of natural
gap). Measured levels on two clips: source fan floor −66..−70 dBFS, quiet
voice segments −59..−64 — a usable gap between them. Threshold ladder
auditioned by ear on kept files + "discarded" (removed-spans-only) audit
streams. Owner verdicts: −48 dB removes quiet voice moments; **−60 dB is
approved** on both clips (discarded streams = fan noise + clicks only,
speech intact). Wire-up: `extract-audio.sh` now trims every new cut
(constants TRIM_DB/TRIM_GAP/TRIM_KEEP = −60/0.25/0.08) and `--retrim`
applies the identical pass to existing wavs idempotently.

**Round-2 change set (dataset-side "silence policy" round):**
1. Apply the approved trim to all selected clips (`--retrim`).
2. Raise the silence anchor: `fine-tune.sh` passes `--include-mutes 10`
   to the extract stage (30 s of true-silence supervision vs the default
   6 s; knob exists to counteract the bed; keep at 2 or omit to isolate
   variables if a round needs it separately).
3. Cutlist growth toward 20+ min remains a separate round (as planned in
   §6/Run-1 next-tune ideas).
Round-2 audition gate, same as Run 1: renditions fed back through the
model, owner listens for the extra-noise defect; note the deleted spans
are the same place clicks were, so a click pass (`adeclick`) may come
later as a separate one-variable round.

**Tool note (same day): rerun bug fixed in extract-audio.sh** — listed
ranges were only counted when *newly cut*, so a rerun with all ranges
already on disk fell through to the whole-video path and cut 5 h of
`_full.wav` spuriously (exposed by the first --retrim run; spurious file
deleted, counter now tracks listed ranges, two clean reruns verified).
Also wired: `fine-tune.sh --include-mutes N` pass-through to the extract
stage (core.py accepts 0–10, default 2).
