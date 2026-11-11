#!/usr/bin/env bash
set -euo pipefail

# fine-tune.sh — fine-tune a pretrained RVC voice model on a voice's
# `selected/` clips, via Applio (https://github.com/IAHispano/Applio).
#
# Usage:
#   scripts/fine-tune.sh <voice> [epochs] [--include-mutes N]
#                                              # default epochs: 100, mutes: 2
#
# --include-mutes N passes Applio's silence-anchor knob through (0-10): how
# many 3 s true-silence samples per speaker get appended to the training
# set. Default 2 = tool default. Raising it teaches "quiet input -> quiet
# output" uncompromisingly; pairs with the quiet-gap trimming in
# extract-audio.sh (see SPEC §6.1).
#
# Requires: ffmpeg+ffprobe, uv, and Applio cloned into external/Applio
# with its venv initialized (see README "Fine-tuning" for the one-time
# setup steps — pure pip/uv, no Homebrew needed).
#
# This is a thin wrapper (SPEC C7: wrap existing tooling, don't reinvent):
# it runs Applio's own preprocess → extract → train pipeline with the exact
# arguments that worked, and writes all outputs under data/<voice>/models/.
# Applio's train step resumes from its latest checkpoint, so you can Ctrl-C
# and rerun safely.

VOICE=""
EPOCHS=""
MUTES=""          # unset → use Applio default (2)
while [[ $# -gt 0 ]]; do
    case "$1" in
        --include-mutes) [[ $# -ge 2 ]] || { echo "--include-mutes needs 0..10" >&2; exit 1; }
                         [[ $2 =~ ^[0-9]+$ ]] && [ "$2" -le 10 ] || { echo "--include-mutes needs 0..10" >&2; exit 1; }
                         MUTES=$2; shift ;;
        *) if [[ -z $VOICE ]]; then VOICE=$1
           elif [[ -z $EPOCHS ]]; then EPOCHS=$1
           else echo "error: extra argument: $1" >&2; exit 1; fi ;;
    esac
    shift
done
[[ -n $VOICE ]] || { echo 'usage: scripts/fine-tune.sh <voice> [epochs] [--include-mutes N]' >&2; exit 1; }
EPOCHS=${EPOCHS:-100}

REPO="$(cd "$(dirname "$0")/.." && pwd)"
MODEL_NAME="$VOICE"
SELECTED="$REPO/data/$VOICE/selected"
MODELS="$REPO/data/$VOICE/models"

[ -d "$SELECTED" ] || { echo "No dataset: $SELECTED missing" >&2; exit 1; }
APPLIO="$REPO/external/Applio"
[ -d "$APPLIO" ] || { echo "Applio missing: $APPLIO (see README)" >&2; exit 1; }

cd "$APPLIO"
# Keep all Applio outputs (training logs, checkpoints, final models)
# inside the project's voice-first layout, out of the repo.
export APPLIO_LOGS_DIR="$MODELS"
PYTHON="$APPLIO/.venv/bin/python"

# Target sample rate (Hz). 40000 = standard quality for speech; 32000 trains
# faster/lower quality. Training at 40k finished in ~40 s/epoch for a 9-min
# dataset on Apple-silicon MPS.
SR=40000

set -x
# Step 1/3: dataset prep (slice, resample, normalize) — delegated to Applio.
"$PYTHON" core.py preprocess --model-name "$MODEL_NAME" \
  --dataset-path "$SELECTED" --sample-rate "$SR" --cpu-cores 6

# Station 2/3: pitch (f0, RMVPE) + contentvec-style embeddings per clip.
MUTE_ARG=()
[[ -n $MUTES ]] && MUTE_ARG=(--include-mutes "$MUTES")
"$PYTHON" core.py extract --model-name "$MODEL_NAME" --f0-method rmvpe \
  --cpu-cores 6 --gpu "-" --sample-rate "$SR" "${MUTE_ARG[@]}"

# Step 3/3: the fine-tune. Resumes from latest checkpoint if any exist.
# `--save-only-latest` keeps disk/IO sane; checkpoints every 25 epochs so
# you can always rewind.
"$PYTHON" core.py train --model-name "$MODEL_NAME" --sample-rate "$SR" \
  --total-epoch "$EPOCHS" --save-every-epoch 25 --save-only-latest \
  --batch-size 8 --gpu 0 --pretrained --index-algorithm Faiss

# Result: MODELS/<voice>/<voice>_<epoch>e_<step>s.pth + .index file under
# MODELS/<voice>/generated/. Voice converter inputs: pth file + index.
