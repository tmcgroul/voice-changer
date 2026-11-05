#!/usr/bin/env bash
set -euo pipefail

# C locale: awk printf must use '.' decimals (ffmpeg -ss fails otherwise)
# and "for (i in a)" ordering is unspecified under some locales/awks.
# Note: this forces byte matching — ASCII fine; cutlist comments may use
# any text, only lines starting with # (after optional whitespace) are skipped.
export LC_ALL=C

# extract-audio.sh — curated voice recordings: download + cut listed time ranges.
#
# Usage:
#   scripts/extract-audio.sh <voice> [cutlist-file] [--demucs]
#                                          [--cookies-from-browser BROWSER]
#
# YouTube may bot-check yt-dlp; --cookies-from-browser chrome/safari/firefox
# uses your browser's cookies in that case and is forwarded to yt-dlp.
#
# The cutlist defaults to data/<voice>/cutlist.txt.
#
# You watch the source video(s) and note the time ranges where the target voice
# speaks cleanly. Write them in the cutlist file like this:
#
#     # cutlist for voice "example" — lines starting with # are ignored
#     https://www.youtube.com/watch?v=XfwjO8OBUw
#     0:34    5:12
#     7:01    12:50
#
#     # a section with NO range lines below it: the whole video is used
#     https://youtu.be/XfwjO8OBUw
#
# Rules:
#   - a bare URL line starts a new video section; the "START END" lines after
#     it belong to that video
#   - a section with no range lines means: process the whole video
#   - timestamps accept HH:MM:SS, MM:SS, or plain seconds
#   - --demucs: strip music/noise (vocals-only) before cutting — use for
#     sources with background music or sound beds; slow but automatic
#
# Output (one folder per voice):
#   data/<voice>/cutlist.txt   the ranges (write this while watching)
#   data/<voice>/raw/          complete downloaded audios (kept for re-cutting)
#   data/<voice>/selected/     one wav per range (<id>_<start>_<end>.wav or
#                              <id>_full.wav) — the folder for RVC WebUI
#                              dataset prep; fine-tune output later goes to
#                              data/<voice>/models/
#
# Behavior notes:
#   - Rerunning is safe: already-downloaded videos and already-cut ranges are
#     skipped; raw downloads are never deleted.
#   - Range cuts supersede whole-video cuts: if a section gains ranges, the
#     stale <id>_full.wav is removed on the next run.
#   - To re-cut a range, delete its wav and run again.

if [[ $# -lt 1 || $# -gt 3 ]]; then
    grep '^#' "$0" | head -n 40 | sed 's/^# \{0,1\}//;s/^#$//'
    exit 1
fi

umask 022

NAME=""
DEMUCS=false
LIST=""
COOKIES=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --demucs) DEMUCS=true ;;
        --cookies-from-browser) COOKIES=(--cookies-from-browser "$2"); shift ;;
        *) if [[ -z "$NAME" ]]; then NAME=$1; elif [[ -z "$LIST" ]]; then LIST=$1
           else echo "error: extra argument: $1" >&2; exit 1; fi ;;
    esac
    shift
done

for tool in yt-dlp ffmpeg ffprobe; do
    command -v "$tool" > /dev/null || { echo "error: '$tool' not found — install it (see README Requirements)" >&2; exit 1; }
done
if $DEMUCS; then
    python3 -c "import demucs" > /dev/null 2>&1 || { echo "error: demucs not installed (pip install demucs)" >&2; exit 1; }
fi

LIST=${LIST:-data/${NAME}/cutlist.txt}
[[ -f $LIST ]] || { echo "error: no cutlist at $LIST (pass a path or create data/${NAME}/cutlist.txt first)" >&2; exit 1; }

RAW="data/${NAME}/raw"
OUT="data/${NAME}/selected"
mkdir -p "$RAW" "$OUT" "data/${NAME}/models"

# "1:02:03" / "02:03" / "3" -> seconds (via awk)
sec() { awk -v t="$1" 'BEGIN{n=split(t,a,":"); r=0; for(i=1;i<=n;i++) r=r*60+a[i]; printf "%.3f", r}'; }

# downloaded file for a video id (any audio extension) or nothing
ext_for() {
    local id=$1 e
    for e in m4a webm mp3 opus ogg wav flac mp4; do
        [[ -f $RAW/$id.$e ]] && { echo "$RAW/$id.$e"; return; }
    done
    return 1
}

# source file to cut from for the current video (respects --demucs)
source_for() {
    local id=$1
    if $DEMUCS; then
        echo "$RAW/${id}_vocals.wav"
    else
        ext_for "$id"
    fi
}

# run demucs on raw/<id>.<ext>, keep vocals as raw/<id>_vocals.wav
demucs_to_vocals() {
    local id=$1 src
    src=$(find "$RAW" -maxdepth 1 -name "$id.*" \! -name "*_vocals.wav" | head -n 1)
    [[ -n $src ]] || { echo "error: no audio file for $id in $RAW" >&2; exit 1; }
    if [[ ! -f $RAW/${id}_vocals.wav ]]; then
        echo "==> separating vocals (demucs) for $id"
        local tmp; tmp=$(mktemp -d)
        python3 -m demucs -n htdemucs --two-stems=vocals --outext wav \
            -o "$tmp" "$src" > /dev/null
        find "$tmp" -name "*.wav" -exec mv {} "$RAW/${id}_vocals.wav" \;
        rm -rf "$tmp"
    fi
}

cur_id=""                 # video id of the section being accumulated
cur_ranges=0              # ranges listed for it so far
cut_files=0; skipped_files=0; total_new=0

# finish processing the current section (called on section change / EOF)
flush_section() {
    [[ -n $cur_id ]] || return 0
    if (( cur_ranges > 0 )); then
        # ranges exist now; drop the stale whole-video cut if any
        rm -f "$OUT/${cur_id}_full.wav" "$OUT/${cur_id}_full.m4a"
    else
        local out="$OUT/${cur_id}_full.wav" src
        src=$(source_for "$cur_id")
        if [[ -f $out ]]; then
            skipped_files=$((skipped_files + 1))
        else
            echo "==> no ranges listed: using whole video $cur_id"
            ffmpeg -hide_banner -loglevel error -i "$src" \
                -vn -acodec pcm_s16le -y "$out" < /dev/null
            cut_files=$((cut_files + 1))
            total_new=$(awk -v s="$total_new" \
                -v d="$(sec "$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$out")")" \
                'BEGIN{printf "%.2f", s+d}')
        fi
    fi
    cur_id=""; cur_ranges=0
}

cut_range() {
    local start=$1 end=$2 id=$cur_id
    s=$(sec "$start"); e=$(sec "$end")
    local duration
    duration=$(awk -v s="$s" -v e="$e" 'BEGIN{printf "%.3f", e - s}' < /dev/null)
    awk -v d="$duration" 'BEGIN{exit (d>0.05)?0:1}' < /dev/null || {
        echo "error: non-positive range $start..$end in $id" >&2; exit 1; }
    local src; src=$(source_for "$id")
    local outfile="$OUT/${id}_${start//:/-.}_${end//:/-.}.wav"
    if [[ -f $outfile ]]; then
        skipped_files=$((skipped_files + 1))
        return
    fi
    ffmpeg -hide_banner -loglevel error \
        -ss "$s" -t "$duration" -i "$src" \
        -vn -acodec pcm_s16le -y "$outfile" < /dev/null
    cur_ranges=$((cur_ranges + 1))
    cut_files=$((cut_files + 1))
    total_new=$(awk -v s="$total_new" -v d="$duration" 'BEGIN{printf "%.2f", s+d}')
}

# NOTE: read on fd 3, and every child below gets </dev/null — otherwise ffmpeg
# or yt-dlp inherit the loop's stdin (= this cutlist) and eat remaining lines.
while read -r line <&3 || [[ -n $line ]]; do
    [[ $line =~ ^[[:space:]]*$ || $line =~ ^[[:space:]]*# ]] && continue

    if [[ $line =~ ^https?:// ]]; then
        flush_section
        vid=""
        [[ $line =~ v=([^&/[:space:]]+) ]] && vid=${BASH_REMATCH[1]}
        # short-circuit: if we can already tell the id from the URL and the raw
        # file is here, skip the network entirely (yt-dlp may be bot-blocked)
        if [[ -z $vid ]] || ! ext_for "$vid" > /dev/null; then
            vid=$(yt-dlp --no-download --no-warnings --print id \
                      "${COOKIES[@]}" "$line" < /dev/null)
            if ! ext_for "$vid" > /dev/null; then
                echo "==> downloading $vid"
                yt-dlp -f "bestaudio[ext=m4a]/bestaudio" --no-warnings \
                    "${COOKIES[@]}"   \
                    -o "${RAW}/%(id)s.%(ext)s" "$line" < /dev/null > /dev/null
                ext_for "$vid" > /dev/null || { echo "error: download produced no audio file for $vid" >&2; exit 1; }
            fi
        fi
        local_id=$vid
        cur_id=$local_id; cur_ranges=0
        if $DEMUCS; then demucs_to_vocals "$local_id"; fi
        continue
    fi

    start=$(awk '{print $1}' <<< "$line")
    end=$(awk '{print $2}' <<< "$line")
    [[ -n ${start:-} && -n ${end:-} ]] || { echo "error: bad range line: $line" >&2; exit 1; }
    [[ -n $cur_id ]] || { echo "error: range line without a preceding URL: $line" >&2; exit 1; }

    cut_range "$start" "$end"
done 3< "$LIST"
flush_section

echo
echo "cut: $cut_files new, $skipped_files already done -> $OUT/"
awk -v t="$total_new" -v c="$cut_files" \
    'BEGIN{ if (c>0) printf "new material this run: %.1f min\n", t/60 }'
printf 'next: drop %s into RVC WebUI "Train a model" -> dataset preparation\n' "$OUT/"
