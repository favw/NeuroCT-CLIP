#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  evaluate_saved_threshold.sh RESULTS_FOLDER LABELS_CSV THRESHOLD

Example:
  ./scripts/evaluate_saved_threshold.sh data/inference_results data/labels.csv 0.30

The results folder must contain labels_weights.npz and predicted_weights.npz.
The script does not rerun the model; it applies THRESHOLD to the saved scores.

Optional environment:
  NEUROCT_PYTHON  Python executable to use (default: python3).
EOF
}

if [[ $# -ne 3 ]]; then
    usage >&2
    exit 2
fi

results_folder=$1
labels_csv=$2
threshold=$3
python_executable=${NEUROCT_PYTHON:-python3}

exec "$python_executable" - "$results_folder" "$labels_csv" "$threshold" <<'PY'
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, hamming_loss


results_folder = Path(sys.argv[1]).expanduser().resolve()
labels_csv = Path(sys.argv[2]).expanduser().resolve()

try:
    threshold = float(sys.argv[3])
except ValueError as exc:
    raise SystemExit(f"THRESHOLD must be a number between 0 and 1: {sys.argv[3]}") from exc

if not 0.0 <= threshold <= 1.0:
    raise SystemExit(f"THRESHOLD must be between 0 and 1: {threshold}")

labels_path = results_folder / "labels_weights.npz"
scores_path = results_folder / "predicted_weights.npz"
for required_path in (labels_path, scores_path, labels_csv):
    if not required_path.is_file():
        raise SystemExit(f"Required file not found: {required_path}")

labels = np.load(labels_path)["data"].astype(int)
scores = np.load(scores_path)["data"]
if labels.shape != scores.shape:
    raise SystemExit(
        f"Label and score shapes differ: labels={labels.shape}, scores={scores.shape}"
    )

label_names = [
    column
    for column in pd.read_csv(labels_csv, nrows=0).columns
    if column != "VolumeName"
]
if len(label_names) != labels.shape[1]:
    raise SystemExit(
        f"Labels CSV has {len(label_names)} pathology columns, but predictions have "
        f"{labels.shape[1]} columns."
    )

predictions = (scores >= threshold).astype(int)
report_text = classification_report(
    labels,
    predictions,
    target_names=label_names,
    digits=4,
    zero_division=0,
)
report_dict = classification_report(
    labels,
    predictions,
    target_names=label_names,
    output_dict=True,
    zero_division=0,
)

summary = (
    f"Decision threshold: {threshold:.6g}\n"
    f"Exact-match accuracy: {accuracy_score(labels, predictions):.4f}\n"
    f"Hamming loss: {hamming_loss(labels, predictions):.4f}\n\n"
    f"{report_text}"
)
threshold_tag = f"{threshold:.6g}".replace(".", "p")
report_path = results_folder / f"classification_report_threshold_{threshold_tag}.txt"
metrics_path = results_folder / f"classification_metrics_threshold_{threshold_tag}.csv"
predictions_path = results_folder / f"binary_predictions_threshold_{threshold_tag}.npz"

report_path.write_text(summary, encoding="utf-8")
pd.DataFrame(report_dict).transpose().to_csv(metrics_path)
np.savez(predictions_path, data=predictions)

print(summary)
print("\nSaved:")
print(report_path)
print(metrics_path)
print(predictions_path)
PY
