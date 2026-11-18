#!/usr/bin/env bash
# Download and extract benchmark datasets into $PIXELMAP_BENCH_DATA
# (default ~/datasets/pixelmap-bench).
#
#   ./download.sh hpatches eth3d      # the wide-baseline sets (~1.3 GB)
#   ./download.sh all                 # adds KITTI 2015 (1.7 GB) and Sintel (5.6 GB)
#
# Archives are deleted after extraction unless KEEP_ARCHIVES=1. Already extracted
# datasets are skipped. Each dataset has its own licence: read it before publishing
# results or derived images.
#   HPatches  https://github.com/hpatches/hpatches-dataset
#   ETH3D     https://www.eth3d.net/datasets
#   KITTI     https://www.cvlibs.net/datasets/kitti/eval_scene_flow.php (non-commercial)
#   Sintel    http://sintel.is.tue.mpg.de/
set -euo pipefail

DATA="${PIXELMAP_BENCH_DATA:-$HOME/datasets/pixelmap-bench}"
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${PYTHON:-$HERE/.venv/bin/python}"
mkdir -p "$DATA"

fetch() {  # url dest
    if [[ ! -s "$2" ]]; then
        echo "downloading $1"
        curl -fL --retry 3 -C - -o "$2.part" "$1"
        mv "$2.part" "$2"
    fi
}

extract_7z() {  # archive dir
    mkdir -p "$2"
    if command -v 7z >/dev/null; then
        7z x -y -o"$2" "$1" >/dev/null
    else
        "$PY" -m py7zr x "$1" "$2"
    fi
}

done_archive() { [[ "${KEEP_ARCHIVES:-0}" == 1 ]] || rm -f "$1"; }

hpatches() {
    [[ -d "$DATA/hpatches-sequences-release" ]] && { echo "hpatches: present"; return; }
    local a="$DATA/hpatches-sequences-release.zip"
    fetch https://huggingface.co/datasets/vbalnt/hpatches/resolve/main/hpatches-sequences-release.zip "$a"
    unzip -q "$a" -d "$DATA"
    done_archive "$a"
}

eth3d() {
    [[ -d "$DATA/eth3d/two_view_training_gt" ]] && { echo "eth3d: present"; return; }
    mkdir -p "$DATA/eth3d"
    for part in two_view_training two_view_training_gt; do
        local a="$DATA/eth3d/$part.7z"
        fetch "https://www.eth3d.net/data/$part.7z" "$a"
        extract_7z "$a" "$DATA/eth3d/$part"
        done_archive "$a"
    done
}

kitti15() {
    [[ -d "$DATA/kitti15/training" ]] && { echo "kitti15: present"; return; }
    local a="$DATA/data_scene_flow.zip"
    fetch https://s3.eu-central-1.amazonaws.com/avg-kitti/data_scene_flow.zip "$a"
    mkdir -p "$DATA/kitti15"
    unzip -q "$a" 'training/image_2/*' 'training/flow_occ/*' 'training/flow_noc/*' -d "$DATA/kitti15"
    done_archive "$a"
}

sintel() {
    [[ -d "$DATA/sintel/training/flow" ]] && { echo "sintel: present"; return; }
    local a="$DATA/MPI-Sintel-complete.zip"
    fetch http://files.is.tue.mpg.de/sintel/MPI-Sintel-complete.zip "$a"
    mkdir -p "$DATA/sintel"
    unzip -q "$a" 'training/clean/*' 'training/final/*' 'training/flow/*' 'training/occlusions/*' -d "$DATA/sintel"
    done_archive "$a"
}

[[ $# -eq 0 ]] && { echo "usage: $0 {hpatches|eth3d|kitti15|sintel|all}..." >&2; exit 2; }
for d in "$@"; do
    case "$d" in
        all) hpatches; eth3d; kitti15; sintel ;;
        hpatches|eth3d|kitti15|sintel) "$d" ;;
        *) echo "unknown dataset $d" >&2; exit 2 ;;
    esac
done
echo "datasets are in $DATA"
