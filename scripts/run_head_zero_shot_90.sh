#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  run_head_zero_shot_90.sh CHECKPOINT DATA_FOLDER REPORTS_CSV LABELS_CSV OUTPUT_FOLDER [TARGET_DEPTH]

Arguments:
  CHECKPOINT     Trained CT-CLIP .pt checkpoint.
  DATA_FOLDER    Preprocessed NIfTI validation/test dataset.
  REPORTS_CSV    Reports CSV used to match volumes.
  LABELS_CSV     Labels CSV containing VolumeName and pathology columns.
  OUTPUT_FOLDER  New folder for predictions and evaluation artifacts.
  TARGET_DEPTH   Optional slice count; defaults to 90 and must be divisible by 10.

Optional environment:
  NEUROCT_PYTHON  Python executable to use (default: python3).
EOF
}

if [[ $# -lt 5 || $# -gt 6 ]]; then
    usage >&2
    exit 2
fi

checkpoint=$1
data_folder=$2
reports_csv=$3
labels_csv=$4
output_folder=$5
target_depth=${6:-90}
python_executable=${NEUROCT_PYTHON:-python3}

for required_file in "$checkpoint" "$reports_csv" "$labels_csv"; do
    if [[ ! -f "$required_file" ]]; then
        echo "Error: file not found: $required_file" >&2
        exit 2
    fi
done

if [[ ! -d "$data_folder" ]]; then
    echo "Error: data folder not found: $data_folder" >&2
    exit 2
fi

if ! [[ "$target_depth" =~ ^[0-9]+$ ]] || (( target_depth == 0 || target_depth % 10 != 0 )); then
    echo "Error: TARGET_DEPTH must be a positive multiple of 10." >&2
    exit 2
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

cd "$script_dir"
exec "$python_executable" run_zero_shot.py \
    --head \
    --preprocessed-nifti \
    --target-depth "$target_depth" \
    --pretrained "$checkpoint" \
    --data-folder "$data_folder" \
    --reports-file "$reports_csv" \
    --labels "$labels_csv" \
    --save "$output_folder"
