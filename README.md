# Voice Changer

Downloadable, open-source real-time voice changer. Built on **RVC**
(Retrieval-based Voice Conversion) with classic DSP around the neural core.
v0 targets **macOS**.

The full picture lives in [SPEC.md](SPEC.md) — goals, architecture, and the
current next step (§7). This file covers the data workflow: preparing voice
data to fine-tune a model.

## Data workflow: from a YouTube video to a fine-tune-ready folder

The scenario: you've found speech of a target voice on YouTube and want the
best possible training material with the least manual work. The pipeline is
two scripted steps: **download + curate**, then hand the result to **RVC's
own tooling** for the rest — slicing, resampling, denoising are delegated,
never reinvented here.

### Data layout (voice-first)

Everything for one voice lives in one folder:

```
data/<voice>/
  cutlist.txt   time ranges you select while watching (curated workflow only)
  raw/          complete downloaded source audio (kept for re-cutting)
  selected/     curated wavs — this folder goes into RVC dataset prep
  models/       fine-tuned .pth output lands here (created empty)
```

Cutlists, recordings and models are git-ignored (see `.gitignore`): they are
large, private, and reproducible from the cutlist + scripts.

### Workflow

**1. Watch and note the cutlist** — open the video and write the ranges where
the target voice speaks cleanly (no music, no other speakers) into
`data/<voice>/cutlist.txt`:

```
# lines starting with # are ignored
https://www.youtube.com/watch?v=XfwjO8OBUw
0:34    5:12
7:01    12:50

# a URL with no ranges below it: the whole video is used
https://youtu.be/XfwjO8OBUw
```

Timestamps accept `HH:MM:SS`, `MM:SS`, or plain seconds. Everything not listed
is dropped — for fully clean sources this means you can list zero ranges.
Aim for **20+ minutes** of kept speech in total (SPEC §6).

**2. Run the extraction**

```bash
./scripts/extract-audio.sh <voice>
```

Downloads the audio and cuts the listed ranges (or whole videos) into
`data/<voice>/selected/`. Reruns are safe — already-cut parts are skipped, so
the cutlist can grow incrementally. Delete a wav and rerun to re-cut it.

For sources with background music, add `--demucs` (once: `pip install demucs`)
to strip vocals-only before cutting.

**3. Hand off to RVC**

In the RVC WebUI → "Train a model" → Step 1, point dataset preparation at the
`selected/` folder (it does slicing, resampling, noise reduction). Spot-check
a few extracted files before training; if the total kept speech is far below
20 minutes, list more ranges rather than training on thin data. Fine-tuned
models end up in `data/<voice>/models/`.

Rule of thumb: whole-video when the source is fully clean; ranges as soon as
music, other speakers, ads, or dead air appear in it.

## Requirements (data workflow)

- **yt-dlp** — downloading the source videos' audio. It's a Python package;
  `pip install yt-dlp` (any venv/`--user`/pipx setup works).
- **ffmpeg + ffprobe** — audio extraction and trimming.

Demucs is only needed when you run the script with `--demucs`
(`pip install demucs`).

The script checks for missing tools and points here when something's absent.

## Ethics / rights

Only fine-tune on voices you have the right to imitate and use. A one-line
attribution or permission note per voice in `data/<voice>/cutlist.txt`
comments is a cheap habit to keep things unambiguous.
