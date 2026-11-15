#!/usr/bin/env bash
set -euo pipefail

# new-voice.sh — create the working folder for a new voice.
#
# Usage:
#   scripts/new-voice.sh <name> [--from <dir>]
#
# Every folder in data/ is one self-sufficient voice, data/<name>/: it holds
# its cutlist, raw sources, selected clips, TRAININGLOG (from its first
# fine-tune run), checkpoints and final model — nothing is shared between
# folders, deleting one deletes everything it needed.
#
# <name> is yours to choose (<voice>, <voice>-clean, <voice>-trimmed-mutes10
# in examples — any folder-safe name). Names carry no structure and imply no
# lineage.
#
# --from <dir>: seed the new folder by cloning cutlist.txt, raw/ and
# selected/ from an existing voice folder (APFS clone on macOS — no extra
# disk until files diverge). Omitted: fresh setup — empty cutlist. A seeded
# folder also gets its sources without re-downloading; a fresh one
# re-downloads whatever its cutlist lists on the first extract run.

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

NAME=""
FROM=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --from) [[ $# -ge 2 ]] || { echo "error: --from needs a folder" >&2; exit 1; }
                FROM=$2; shift ;;
        -h|--help) grep '^#' "$0" | sed '1d' | head -n 30 | sed 's/^# \{0,1\}//;s/^#$//'; exit 0 ;;
        *) if [[ -z $NAME ]]; then NAME=$1
           else echo "error: extra argument: $1" >&2; exit 1; fi ;;
    esac
    shift
done
[[ -n $NAME ]] || { echo 'usage: scripts/new-voice.sh <name> [--from <dir>]' >&2; exit 1; }
[[ $NAME =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]] || { echo "error: bad voice name: $NAME" >&2; exit 1; }

if [[ -n $FROM ]]; then
    FROM=${FROM%/}; FROM=${FROM#./}; FROM=${FROM#data/}
    [[ -d data/$FROM ]] || { echo "error: --from: no such folder: data/$FROM" >&2; exit 1; }
    FROM="data/$FROM"
    [[ -f $FROM/cutlist.txt ]] || { echo "error: --from: $FROM has no cutlist.txt" >&2; exit 1; }
fi

DIR="data/$NAME"
[[ ! -e $DIR ]] || { echo "error: $DIR already exists" >&2; exit 1; }

# clone <src> <dst> — APFS clone on macOS (no extra disk until divergence),
# plain recursive copy elsewhere.
clone() {
    local src=$1 dst=$2
    if ! cp -cR "$src" "$dst" 2>/dev/null; then
        rm -rf "$dst"
        cp -R "$src" "$dst"
    fi
}

mkdir -p "$DIR"
if [[ -n $FROM ]]; then
    clone "$FROM/cutlist.txt" "$DIR/cutlist.txt"
    [[ -d $FROM/raw ]] && clone "$FROM/raw" "$DIR/raw"
    [[ -d $FROM/selected ]] && clone "$FROM/selected" "$DIR/selected"
    echo "seeded from $FROM (clones — no extra disk on APFS)"
else
    cat > "$DIR/cutlist.txt" <<EOF
# cutlist for voice "$NAME" — lines starting with # are ignored
#
# A bare URL line starts a video section; the "START END" lines after it
# are the kept ranges. A section with no range lines uses the whole video.
#
#     https://www.youtube.com/watch?v=XfwjO8OBUw
#     0:34    5:12
EOF
fi

echo "created $DIR"
