# Voice Changer — Project Specification (v0.3)

> Status: working spec (v0.3, 2026-11-05). Living document — updated as decisions are made.
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

#### 4.2.1 Measured chunk latency (2026-11-05, M3 Pro / 36 GB / macOS 14.6.1, gvs_100e_2700s.pth, 40 kHz v2)

Chunked per-stage timing through Applio's own inference paths
(`scripts/bench-rvc.py`, n=25 chunks/row, p95 after 5 warm-up chunks;
full JSON in `data/gvs/benchmarks/`). Context = audio padded on both sides for
the f0/embedding pass (same op their whole-file path performs):

| window | context | RMVPE f0 | ContentVec | faiss | generator | total p95 | RTF | est. e2e* |
|---|---|---|---|---|---|---|---|---|
| 0.1 s | ±0.25 s | 22 ms | 8 ms | 3 ms | 37 ms | **69 ms** | 0.69 | ~190–220 ms |
| 0.25 s | ±0.25 s | 27 ms | 9 ms | 3 ms | 43 ms | **81 ms** | 0.32 | ~330–380 ms |
| 0.5 s | ±0.25 s | 29 ms | 10 ms | 4 ms | 51 ms | **92 ms** | 0.18 | ~440–470 ms |
| 1.0 s | ±0.25 s | 33 ms | 13 ms | 5 ms | 70 ms | **120 ms** | 0.12 | ~570 ms |
| any | ±0.5 s | — | — | — | — | +20–30 ms vs row above | — | — |

\* e2e ≈ window + compute + capture/DSP/output buffers (30 + 25 ms) — not yet
measured with a live audio loop; treat as design estimate.

**Verdict (MPS on M3 Pro):**
- Sustained chunked inference is comfortably real-time (RTF 0.12–0.69, no
  model-load cost once warm — the earlier 0.6–0.8× whole-file figure loaded
  the model + ran a full-file f0 pass inside the measurement).
- The **150–200 ms target is only reachable with 0.1 s windows**; 0.25 s
  chunks already put e2e near/above the 300 ms ceiling. The old §4.2 preset
  table implicitly assumed ~20–30 ms capture; that is wrong for a chunked
  core — chunk accumulation time is the window size.
- **MPS hard limit:** the HiFi-GAN decoder emits audio length as conv1d
  output channels and MPS rejects > 65536 channels → **≤ ~1.64 s of audio
  per generator call at 40 kHz**. Applio's whole-file padding (t_pad = ±1.0 s)
  therefore fails on MPS at every chunk size (0.1–1.0 s), and ±0.5 s context
  kills the 1.0 s window. The C3 chunker must keep window + 2·context
  ≤ ~1.5 s; slimmer context also saves 20–30 ms of compute per chunk.
- Cost anatomy: RMVPE and the generator scale with window size (incl. context);
  faiss and ContentVec are nearly flat; larger context pads inflate f0/
  embed/gen proportionally (±0.5 s context vs ±0.25 s ≈ +20–30 ms total).
- Generous CPU headroom remains (faiss + RMVPE preprocessing are
  single-threaded on macOS by Applio's OMP_NUM_THREADS=1 default).

Preset mapping to adopt (revised from measurements):
| Preset | Window | Context | Est. e2e | Note |
|---|---|---|---|---|
| Fast | 0.1 s | ±0.25 s | ~200–220 ms | meets target zone; under 300 ms ceiling |
| Balanced | 0.25 s | ±0.25 s | ~380 ms | above old 300 ms ceiling — trim via context/optimization or accept |
| Quality | 0.5–1.0 s | ±0.25 s | ~460–570 ms | streaming/relaxed use only |

Open before committing the Fast preset: **0.1 s windows are unmeasured for
quality** (f0 smoother + index retrieval get less audio to work with) —
audition before locking.

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

*All data for one voice lives in one folder* (voice-first layout):
`data/<voice>/` with `cutlist.txt` (curated time ranges), `raw/` (full source
downloads, kept for re-cutting), `selected/` (curated wavs — the folder that goes
into RVC WebUI dataset prep), `models/` (fine-tuned .pth output),
`benchmarks/` (voice-specific measurement outputs, e.g. §4.2.1 latency
JSONs), `audition/` (models + renditions used for hearing tests). The workflow is
driven by one script + its cutlist (`scripts/extract-audio.sh`): a URL line starts
a video section; the `START END` lines below it are the kept ranges; a section
with no ranges uses the whole video. `--demucs` strips music/noise first for
sources that need it. Quality criteria: solo speech, decent audio quality.
Choosing source material / ranges is the only manual step; everything after
(slicing, resampling, denoising) is delegated to RVC's dataset prep.

**Current next step:** chunked MPS latency is measured (§4.2.1): real-time
viable, but Fast-preset viability hinges on **auditioning 0.1 s windows for
quality** (f0 smoother and index retrieval degrade with short chunks —
unmeasured so far) before locking the window size, then writing the **C3
chunker design doc** (window/context/crossfade choices plus the MPS
≤ 65536-sample decoder limit). CPU/MPS decisiveness is settled; per-platform
work may start after that. Cutlist growth past 20 min (§6) stays a parallel
task for a later comparison run.
