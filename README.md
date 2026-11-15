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

### Data layout: one folder per voice

Every folder in `data/` is one self-sufficient voice — it holds everything
that produced it, shares nothing, and deleting it deletes its entire history
(SPEC §6):

```
data/<voice>/
  cutlist.txt    time ranges you select while watching (curated workflow only)
  raw/           complete downloaded source audio (kept for re-cutting)
  selected/      curated wavs — this folder goes into RVC dataset prep
  TRAININGLOG.md provenance record (one entry per run, written by fine-tune.sh)
  <voice>_<N>e_<step>s.pth + .index      the fine-tuned model
  checkpoints/   Applio workspace (extracted features, G/D checkpoints, ...)
```

The voice name is yours to choose — `<voice>`, `<voice>-clean`,
`<voice>-trimmed-mutes10` in the examples below; it carries no structure and
implies no lineage. (Several folders may exist for one voice — a descriptor
suffix distinguishes them.) Start a new voice from scratch, or seed it from
an existing one when the dataset continues (`new-voice.sh --from` — an APFS
clone on macOS: no extra disk until files diverge). Superseded voice folders
are deleted manually after auditioning.

Cutlists, recordings and models are git-ignored (see `.gitignore`): they are
large, private, and reproducible from the cutlist + scripts.

### Workflow

**1. Create a voice folder**

```bash
./scripts/new-voice.sh <name>              # fresh: empty cutlist
./scripts/new-voice.sh <name> --from data/<existing-voice>
```

**2. Watch and note the cutlist** — open the video and write the ranges where
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

**3. Run the extraction**

```bash
./scripts/extract-audio.sh <voice>    # the voice folder you're working on
```

Downloads the audio and cuts the listed ranges (or whole videos) into
`data/<voice>/selected/`. Reruns are safe — already-cut parts are
skipped, so the cutlist can grow incrementally. Delete a wav and rerun to
re-cut it.

For sources with background music, add `--demucs` (once: `pip install demucs`)
to strip vocals-only before cutting.

**4. Fine-tune a voice model**

```bash
./scripts/fine-tune.sh <voice> [epochs]
```

Runs Applio's own pipeline (slice/resample → f0+embedder extraction →
fine-tune) with the settings that worked here. Safe to Ctrl-C and rerun:
training resumes from the latest checkpoint. After each completed run the
script appends a provenance entry to `data/<voice>/TRAININGLOG.md`,
pre-filling Tool, Dataset and Train settings — **fill in the blank fields
(Base model, Timing, Loss track)**, using this field list:

```
Base model:       pretrained pair + rate + vocoder, checkpoint source
Tool:             Applio commit, python, torch, repo-state notes (e.g. patches)
Dataset:          source, #clips, total minutes, sr/channels, trimming, mutes
Train settings:   epochs, batch, f0 method, device, index algorithm, save policy
Timing:           preprocess / extract / train per-epoch + total
Loss track:       start → best@epoch
Checkpoints kept: paths
Output model:     final pth + index path
```

The result —
`data/<voice>/<voice>_<N>e_<step>s.pth` plus the `.index` file in
the same folder — is a standard RVC model: load the pair into any
RVC-compatible tool for conversion.

## Requirements

Everything the scripts need, with the recommended install for each. The
scripts check for missing tools and point here when something's absent.

- **ffmpeg + ffprobe** — audio extraction and trimming.
- **yt-dlp** — downloading the source videos' audio. A Python package:
  `pip install yt-dlp` (any venv/`--user`/pipx setup works).
- **uv** — fast Python venv/pip frontend used to set up the training tool.
  `curl -LsSf https://astral.sh/uv/install.sh | sh` (or `brew install uv`).
- **Applio** — the training tool
  ([IAHispano/Applio](https://github.com/IAHispano/Applio)): actively
  maintained RVC fork with a CLI; setup is pure pip/uv — **no Homebrew build
  steps needed on macOS**, unlike the original RVC WebUI.

  ```bash
  git clone --depth 1 https://github.com/IAHispano/Applio external/Applio
  cd external/Applio
  uv venv .venv --python 3.12
  uv pip install -r requirements.txt      # macOS: torch CPU+MPS wheels
  .venv/bin/python core.py prerequisites --no-exe   # ~2 GB pretrained G/D + predictors
  ```

Demucs is only needed when you run `extract-audio.sh` with `--demucs`
(`pip install demucs`).

## Ethics / rights

Only fine-tune on voices you have the right to imitate and use. A one-line
attribution or permission note per voice in `data/<voice>/cutlist.txt`
comments is a cheap habit to keep things unambiguous.
