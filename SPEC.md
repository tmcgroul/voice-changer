# Voice Changer — Project Specification (v0.2)

> Status: working spec (v0.2, 2026-11-05). Living document — updated as decisions are made.
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
| Capture buffer | 20–30 ms |
| DSP front-end | < 5 ms |
| Neural inference chunk | 100–150 ms (quality/latency knob) |
| DSP back-end + output buffer | 20–30 ms |

Quality presets: Fast (~150 ms) / Balanced (~250 ms) / Quality (~400 ms, streaming only).

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
into RVC WebUI dataset prep), `models/` (fine-tuned .pth output). The workflow is
driven by one script + its cutlist (`scripts/extract-audio.sh`): a URL line starts
a video section; the `START END` lines below it are the kept ranges; a section
with no ranges uses the whole video. `--demucs` strips music/noise first for
sources that need it. Quality criteria: solo speech, decent audio quality.
Choosing source material / ranges is the only manual step; everything after
(slicing, resampling, denoising) is delegated to RVC's dataset prep.

**Current next step:** fine-tuning the pretrained RVC model is done
(first run 2026-11-05, "Creating a voice" flow §6; verdict acceptable as
trained). Next: measure chunked inference latency on MPS against the §4.2
budget — decides CPU/MPS viability before per-platform work starts.
Cutlist growth past 20 min (§6) stays a parallel task for a later
comparison run.
