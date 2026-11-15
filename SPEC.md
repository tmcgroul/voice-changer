# Voice Changer — Project Specification (v0.6)

> Status: working spec (v0.6, 2026-11-15). Living document — updated as decisions are made.
> Some product aspects are intentionally not yet defined (owner will specify later);
> those are marked **[TBD]**. The TBDs do not block the current next step (§7).

## 1. Overview

A downloadable, open-source voice changer: the user installs it, runs a few
setup commands, and ends up with a working real-time voice changer suitable
for any purpose. Cross-platform (Windows, macOS, Linux) is a project goal;
v0 targets macOS.

Technology: **RVC** (Retrieval-based Voice Conversion) — chosen deliberately
because no known alternative offers a better quality/performance/community
tooling combination for per-voice real-time conversion. Classic DSP surrounds
the neural core for signal conditioning.

### Product vision
The value of the project is the *path of least resistance* from "I found this
on GitHub" to "my voice is changed in Discord": a few commands to install and
set up (download a model or train one), system-wide output — without the
commercial product's bloat or subscription.

### Explicitly open / to be defined
- GUI vs CLI vs both **[TBD]**
- Whether training happens locally, guided via cloud notebooks, or both **[TBD]**
- Distribution format and model packaging **[TBD]**

## 2. Goals

- G1. **Frictionless setup:** install + a few commands → working voice changer
- G2. Real-time voice conversion with end-to-end latency < 300 ms (target 150–200 ms)
- G3. System-wide output — works with Discord/Zoom/games without per-app setup
- G4. Per-voice models via fine-tuning (RVC); guided training flow
- G5. Voice library: manage multiple models, switch quickly
- G6. Runs comfortably on a mid-range machine (single core + small GPU slice)

## 3. Target user & use cases

- **v0 platform: macOS** (Windows/Linux planned — see §5)
- UC1: Live calls with a converted voice (<300 ms latency)
- UC2: Streaming/recording with relaxed latency but higher quality

## 4. Architecture

```
┌─────────────┐   ┌──────────────────────────────┐   ┌──────────────────┐
│ Microphone  │──▶│        DSP front-end         │──▶│  Neural core     │
│ (capture)   │   │ noise gate · denoise · EQ    │   │ (RVC inference,  │
└─────────────┘   └──────────────────────────────┘   │  chunked)        │
                                                     └────────┬─────────┘
┌──────────────────────────────────────────────────────────────┐ │
│                     Output layer                             │ ▼
│  ┌────────────────────────┐    ┌──────────────────────────┐  │
│  │ Virtual mic device     │◀───│ DSP back-end             │◀─┘
│  │ (system-wide, G3)      │    │ limiter · gain · EQ      │
│  └────────────────────────┘    └──────────────────────────┘
└──────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│  App shell: UI · voice library · model manager · training    │
│  launcher (delegates to RVC tooling) · settings · hotkeys    │
└──────────────────────────────────────────────────────────────┘
```

### 4.1 Components

| # | Component | Responsibility | Notes |
|---|-----------|----------------|-------|
| C1 | Audio I/O | Capture mic, low-latency buffers (e.g. 128–512 samples) | Platform audio API |
| C2 | DSP front-end | Noise gate, light denoise, normalization before the model | Classic DSP; protects model from junk |
| C3 | Neural core | Chunked RVC/voice-conversion inference; model loading, hot-swap | The only ML part; runs local |
| C4 | DSP back-end | Gain, limiter, optional formant/EQ polish | Classic DSP |
| C5 | Virtual mic | System-wide fake input device other apps select | **Per-platform implementations; main platform risk** |
| C6 | Voice library | Import/manage/switch .pth+index models; metadata | |
| C7 | Training flow | Guided fine-tune: data prep → cloud/local GPU → import result | Wraps existing RVC tooling, do not reinvent |
| C8 | App shell / UI | Glue, settings, hotkeys, latency/quality presets | |

### 4.2 Latency budget (target: 150–200 ms end-to-end)

| Stage | Budget |
|---|---|
| Capture buffer + chunk accumulation | 100–250 ms — **equals the window size itself** (the model can't run on an incomplete chunk; do not budget this as 20–30 ms) |
| DSP front-end | < 5 ms |
| Neural inference compute | 65–125 ms on M3 Pro MPS (measured, window-size dependent) |
| DSP back-end + output buffer | 20–30 ms |

#### 4.2.1 Measured chunk latency (2026-11-05, M3 Pro MPS, a fine-tuned 40 kHz model)

Measured via Applio's own inference paths (`scripts/bench-rvc.py`, n=25
chunks/row, p95 after warm-up; raw JSON in `data/<voice>/benchmarks/`):

| window | compute p95 | est. e2e* |
|---|---|---|
| 0.1 s | 69 ms | ~200–220 ms |
| 0.25 s | 81 ms | ~380 ms |
| 0.5 s | 92 ms | ~470 ms |
| 1.0 s | 120 ms | ~570 ms |

\* e2e ≈ window + compute + I/O buffers — design estimate; not yet measured
with a live audio loop.

**Verdict (MPS on M3 Pro):**
- Sustained chunked inference is comfortably real-time (RTF 0.12–0.69);
  the earlier 0.6–0.8× whole-file figure included model load + a full-file
  f0 pass and no longer applies.
- The **150–200 ms target is only reachable with 0.1 s windows**; 0.25 s
  chunks already put e2e at/above the 300 ms ceiling — chunk accumulation
  time is the window size, not the §4.2 table's 20–30 ms.
- **MPS hard limit:** generator output ≤ 65536 samples → **≤ ~1.64 s of
  audio per call at 40 kHz**; the C3 chunker must keep window + 2·context
  ≤ ~1.5 s. Applio's ±1.0 s context padding fails on MPS at every chunk
  size; slim context also saves 20–30 ms of compute per chunk.

Preset mapping, revised from measurements (context ±0.25 s):
| Preset | Window | Est. e2e | Note |
|---|---|---|---|
| Fast | 0.1 s | ~200–220 ms | under the 300 ms ceiling |
| Balanced | 0.25 s | ~380 ms | above the old ceiling — optimize or accept |
| Quality | 0.5–1.0 s | ~460–570 ms | streaming/relaxed use only |

Open before committing Fast: **0.1 s windows are unmeasured for quality**
(f0 smoother + index retrieval get less audio) — audition before locking.

#### 4.2.2 Live smoke test (2026-11-07): real-time confirmed, engine adopted

A live browser session through Applio's existing realtime tab (streaming
worker on MPS, FP32, with CPU fallback) produced usable live conversion,
equal-or-near-equal quality at 100 vs 250 ms chunks, and slightly robotic
timbre. Consequences:

- **C3 decision: adopt/wrap Applio's realtime engine** (worker, SOLA +
  phase-vocoder crossfade, VAD, hot-swap, MPS fallback) instead of building
  a chunker from scratch — same reuse stance as C7.
- Speed is settled; **quality is the frontier**: the robotic tone belongs to
  the fine-tune stage (next milestone), not the real-time plumbing.
- The naive edge-stitched audition renderer's harsh short-window verdicts
  are partly stitching artifacts; controlled A/B fallback only.
- Presets stay provisional until a scripted loopback latency measurement
  lands (deferred; the 100 ms chunk knob already reduced perceived delay).

## 5. Platform strategy

Cross-platform (Windows, macOS, Linux) is a project goal, but **v0 targets macOS only**
(others planned, order TBD).

To keep later ports cheap, two rules apply from the start:

- The real-time core (C1–C4, C6, C8) is written against a platform-neutral
  audio I/O layer — no macOS-specific code outside the platform layer.
- The virtual mic (C5) hides behind one interface, so a Windows/Linux
  implementation can be added without touching the core.

## 6. How voices work

The project does not create voices from scratch. It uses the pretrained RVC
model — already trained on speech in general — and adapts it to specific voices:

- **Using a voice:** the user picks an existing RVC voice model (trained by
  the community or shared by someone) and the app converts their voice into it.
- **Creating a voice:** the user feeds the app examples of a target voice
  (recordings of the person they want to imitate). The app fine-tunes the
  pretrained model on those examples, producing a new voice model.
  Realistic minimum: ~10–15 minutes of clean recordings; 30–60 minutes
  gives noticeably more consistent results (to be validated in the first fine-tune run).

All voice models are interchangeable files: create one, import one, share it,
use it in real time.

**Voice folder: one self-sufficient folder per voice (2026-11-15 decision)**

Every folder in `data/` is one voice and holds everything that produced it —
model, data, provenance. Folders share nothing; deleting one deletes
everything it needed. Names are freeform and user-chosen (`<voice>`,
`<voice>-clean` in examples) and carry no structure — no versioning, no
implied lineage, no ordering. A user picks a voice, not a version; several
folders for one voice are just different takes, distinguished by name:

- `cutlist.txt`, `raw/`, `selected/` — curated ranges, full downloads (kept
  for re-cutting), dataset wavs for Applio prep
- `TRAININGLOG.md` — provenance-only record, one entry per run written by
  `fine-tune.sh` (field list in README); goes with the folder
- checkpoints (needed for resume) + final pth/index; `benchmarks/`,
  `audition/` (cross-voice comparisons go in the newest folder)

Rules:

- A new voice continues from an existing one (`new-voice.sh <name> --from
  <dir>` clones its cutlist/raw/selected — APFS clone, no extra disk until
  files diverge) or starts fresh. No fork semantics, no `Based on` tracking.
- Retention is manual — superseded folders are deleted by the user after
  auditioning; nothing is pruned automatically.
- Workflow: `new-voice.sh` → watch + write cutlist → `extract-audio.sh` →
  `fine-tune.sh` trains in place, resuming on rerun. Choosing sources and
  ranges is the only manual step (criteria: solo speech, decent audio); the
  rest is RVC dataset prep. Cutlist format, `--demucs`, mutes: README.

Run 1 and Run 2 predate this layout; exact local names live in their
TRAININGLOGs, not in this repo.

#### 6.1 Noise strategy for fine-tuning data (2026-11-09 decisions)

Diagnosed: the fine-tuned model emits a learned "fan noise" floor during
silence — it reproduces the source recordings' constant broadband floor even
on digital-silence input, and the stock base pretrain emits a louder bed
still (A/B via `scripts/noise-floor-report.py`; A/B closed: index-rate and
epoch count are not noise levers). Noise removal by denoising was auditioned
by the owner on real clips and **rejected** — every denoiser variant
(afftdn, RNNoise ward/soft, hybrid) either left noise or damaged the voice.
Decisions:

- **Quiet-gap trimming replaces denoising** as the dataset-side tool:
  `extract-audio.sh` removes quiet spans (RMS < −60 dBFS for ≥0.25 s,
  keeping 0.08 s gaps) from every cut — 2026-11-09 listener-approved on
  two clips. Fan-floor "cold modality supervision" for the model comes
  from pause-heavy material; trimming it out — not denoising the entire
  audio — is the strategy. `--retrim` re-applies to existing wavs.
- **Clean silence supervision for the model** — the mute-anchor knob:
  `--include-mutes 10` at the extract stage (Applio's silence anchor,
  30 s instead of default 6 s) — used in Run 2. The anchor is the only
  training sample whose ground-truth output is true digital silence;
  it is a companion to trimming, not a standalone fix (Run 2 verdict:
  bed persisted).
- The architectural noise floor (pretrain bed) is accepted as-is; no
  inference-side tool can remove what the dataset was trained on.

**Current next step:** real-time viability is settled (§4.2.1 bench +
§4.2.2 live smoke test; C3 = adopted Applio realtime engine). **Run 2
trained** (silence policy per §6.1: trimmed dataset + include-mutes 10;
Run-2's output model and Run-1's model + dataset snapshot both migrated
into their own per-voice folders per §6 — exact local names in their
TRAININGLOGs);
owner verdicts in: no audible noise improvement, no regression; parameter
matrix winner f0=fcpe (protect/rms values pending).
Later rounds (queued): cutlist growth toward 20+ min; click pass
(`adeclick`) as its own round if still needed. Parallel later tasks:
scripted loopback latency measurement to lock the §4.2 presets; C5
virtual mic (system-wide output) when audio-shell work starts.
