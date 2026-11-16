#!/usr/bin/env bash
set -euo pipefail

# C locale: awk printf must use '.' decimals (duration sum in the
# TRAININGLOG entry).
export LC_ALL=C

# fine-tune.sh — fine-tune a pretrained RVC voice model on a voice
# folder's `selected/` clips, via Applio (https://github.com/IAHispano/Applio).
#
# Usage:
#   scripts/fine-tune.sh <voice-folder> [epochs] [--include-mutes N]
#                                        # default epochs: 100
#
# The argument is a voice folder (data/<voice>/, see
# scripts/new-voice.sh), given as its name (<voice>) or path.
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
# arguments that worked, and writes all outputs inside the voice folder
# itself — it owns its inputs and outputs, nothing is shared. Applio's
# train step resumes from its latest checkpoint, so you can Ctrl-C and
# rerun safely.

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
[[ -n $VOICE ]] || { echo 'usage: scripts/fine-tune.sh <voice-folder> [epochs] [--include-mutes N]' >&2; exit 1; }
EPOCHS=${EPOCHS:-100}

REPO="$(cd "$(dirname "$0")/.." && pwd)"
# normalize a voice-folder argument: <voice> or data/<voice> -> data/<voice>
ARG=${VOICE%/}; ARG=${ARG#./}; ARG=${ARG#data/}
DIR="data/$ARG"
[[ -d $DIR ]] || { echo "error: no voice folder $DIR — create one with scripts/new-voice.sh" >&2; exit 1; }
MODEL_NAME=$(basename "$DIR")
SELECTED="$REPO/$DIR/selected"

[ -d "$SELECTED" ] || { echo "No dataset: $SELECTED missing" >&2; exit 1; }
APPLIO="$REPO/external/Applio"
[ -d "$APPLIO" ] || { echo "Applio missing: $APPLIO (see README)" >&2; exit 1; }

cd "$APPLIO"
# Keep all Applio outputs (training logs, checkpoints, final models) inside
# the voice folder, out of the repo: logs root data/ + folder name
# <voice> -> data/<voice>/.
export APPLIO_LOGS_DIR="$REPO/data"
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
# ${MUTE_ARG[@]+...} guards empty-array expansion under bash 3.2 + set -u
"$PYTHON" core.py extract --model-name "$MODEL_NAME" --f0-method rmvpe \
  --cpu-cores 6 --gpu "-" --sample-rate "$SR" \
  ${MUTE_ARG[@]+"${MUTE_ARG[@]}"}

# Step 3/3: the fine-tune. Resumes from latest checkpoint if any exist.
# `--save-only-latest` keeps disk/IO sane; checkpoints every 25 epochs so
# you can always rewind.
"$PYTHON" core.py train --model-name "$MODEL_NAME" --sample-rate "$SR" \
  --total-epoch "$EPOCHS" --save-every-epoch 25 --save-only-latest \
  --batch-size 8 --gpu 0 --pretrained --index-algorithm Faiss

# Result: <DIR>/<voice>_<epoch>e_<step>s.pth + .index file in the same
# folder. Voice converter inputs: pth file + index.

# log_run — provenance: create-or-append TRAININGLOG.md, one entry per
# completed run. Auto-fills what the script knows; leaves Base model,
# Timing, Loss track blank for the user. No verdicts or audition notes.
log_run() {
    local LOG="$REPO/$DIR/TRAININGLOG.md"
    if [[ ! -f $LOG ]]; then
        cat > "$LOG" <<EOF
# Training provenance — $(basename "$DIR")

One entry per fine-tune run, written by fine-tune.sh when the run finishes;
this file belongs to this voice directory. Field list lives in README
("Fine-tune a voice model"). Provenance only — no verdicts or audition notes.

EOF
    fi
    local CLIPS MINUTES COMMIT
    CLIPS=$(find "$SELECTED" -name '*.wav' | wc -l | tr -d ' ')
    MINUTES=$(find "$SELECTED" -name '*.wav' -print0 |
        xargs -0 -n1 ffprobe -v error -show_entries format=duration \
            -of csv=p=0 2>/dev/null | awk '{s+=$1} END {printf "%.1f", s/60}')
    COMMIT=$(git -C "$APPLIO" rev-parse --short HEAD 2>/dev/null || echo '?')
    cat >> "$LOG" <<EOF

### Run: $(basename "$DIR") — $(date +%Y-%m-%d)

- **Base model:**
- **Tool:** Applio $COMMIT (external/Applio), .venv python, macOS/MPS
- **Dataset:** $CLIPS clips, $MINUTES min (selected/), sr $SR mono$( [[ -n $MUTES ]] && echo ", --include-mutes $MUTES" || echo ", mutes default (2)")
- **Train settings:** $EPOCHS epochs, batch 8, save-every 25 (latest only), f0 rmvpe, index Faiss, gpu "-"
- **Timing:**
- **Loss track:**
- **Checkpoints kept:** $REPO/$DIR/checkpoints/
- **Output model:** $REPO/$DIR/$(basename "$DIR")_${EPOCHS}e_*.pth + .index
EOF
    echo "logged run in $LOG (fill in Base model, Timing, Loss track)"
}

log_run

# Standard audition: render the fixed sample through the fresh model so
# every run produces comparable candidates. Fixed settings for all runs
# (whole-file pipeline, script-default index-rate/protect); only f0 method
# differs (rmvpe + fcpe). The sample (23.4 s) exceeds the MPS whole-pass
# limit (SPEC §4.2.1), so renders run on CPU. Failure here must not fail
# the fine-tune run itself.
SAMPLE="$REPO/voice-sample.wav"
if [[ ! -f $SAMPLE ]]; then
    echo "note: no $REPO/voice-sample.wav — standard audition skipped"
else
    PTH=$(ls "$REPO/$DIR/${MODEL_NAME}_${EPOCHS}e_"*.pth 2>/dev/null | head -n1 || true)
    IDX="$REPO/$DIR/${MODEL_NAME}.index"
    if [[ -z $PTH || ! -f $IDX ]]; then
        echo "warning: final model not found — standard audition skipped"
    else
        mkdir -p "$REPO/$DIR/audition"
        for F0 in rmvpe fcpe; do
            OUT="$REPO/$DIR/audition/rend-sample-$F0.wav"
            if ! "$PYTHON" "$REPO/scripts/render-audition.py" \
                --src "$SAMPLE" --mode whole --out "$OUT" \
                --model "$PTH" --index "$IDX" --f0-method "$F0" \
                --device cpu; then
                echo "warning: audition render ($F0) failed — see output above"
            fi
        done
        echo "standard audition: $REPO/$DIR/audition/rend-sample-{rmvpe,fcpe}.wav"
    fi
fi

echo "done"
