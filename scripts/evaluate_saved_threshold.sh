#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  # Apply one manually selected threshold to an inference result:
  evaluate_saved_threshold.sh RESULTS_FOLDER LABELS_CSV THRESHOLD

  # Select one F1-optimal threshold per label on validation data, then apply
  # those thresholds to a separate test result:
  evaluate_saved_threshold.sh --calibrate-f1 VALIDATION_RESULTS TEST_RESULTS LABELS_CSV

Examples:
  ./scripts/evaluate_saved_threshold.sh data/inference_results data/labels.csv 0.30
  ./scripts/evaluate_saved_threshold.sh --calibrate-f1 \
      data/validation_inference data/test_inference data/labels.csv

Each results folder must contain labels_weights.npz and predicted_weights.npz.
The script does not rerun the model.

Optional environment:
  NEUROCT_PYTHON  Python executable to use (default: python3).
EOF
}

python_executable=${NEUROCT_PYTHON:-python3}

if [[ ${1:-} == "--calibrate-f1" ]]; then
    if [[ $# -ne 4 ]]; then
        usage >&2
        exit 2
    fi
    mode=calibrate-f1
    first_results_folder=$2
    second_results_folder=$3
    labels_csv=$4
    mode_value=""
elif [[ $# -eq 3 ]]; then
    mode=fixed
    first_results_folder=$1
    second_results_folder=""
    labels_csv=$2
    mode_value=$3
else
    usage >&2
    exit 2
fi

exec "$python_executable" - \
    "$mode" \
    "$first_results_folder" \
    "$second_results_folder" \
    "$labels_csv" \
    "$mode_value" <<'PY'
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    hamming_loss,
    precision_recall_curve,
)


mode = sys.argv[1]
first_results_folder = Path(sys.argv[2]).expanduser().resolve()
second_results_folder = (
    Path(sys.argv[3]).expanduser().resolve() if sys.argv[3] else None
)
labels_csv = Path(sys.argv[4]).expanduser().resolve()
mode_value = sys.argv[5]


def load_predictions(results_folder):
    labels_path = results_folder / "labels_weights.npz"
    scores_path = results_folder / "predicted_weights.npz"
    for required_path in (labels_path, scores_path):
        if not required_path.is_file():
            raise SystemExit(f"Required file not found: {required_path}")

    labels = np.load(labels_path)["data"].astype(int)
    scores = np.load(scores_path)["data"]
    if labels.shape != scores.shape:
        raise SystemExit(
            f"Label and score shapes differ in {results_folder}: "
            f"labels={labels.shape}, scores={scores.shape}"
        )
    return labels, scores


def validate_label_names(column_count):
    if not labels_csv.is_file():
        raise SystemExit(f"Required file not found: {labels_csv}")
    label_names = [
        column
        for column in pd.read_csv(labels_csv, nrows=0).columns
        if column != "VolumeName"
    ]
    if len(label_names) != column_count:
        raise SystemExit(
            f"Labels CSV has {len(label_names)} pathology columns, but predictions "
            f"have {column_count} columns."
        )
    return label_names


def write_evaluation(results_folder, labels, scores, thresholds, name):
    predictions = (scores >= thresholds[None, :]).astype(int)
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
        f"Threshold mode: {name}\n"
        f"Exact-match accuracy: {accuracy_score(labels, predictions):.4f}\n"
        f"Hamming loss: {hamming_loss(labels, predictions):.4f}\n\n"
        f"{report_text}"
    )
    report_path = results_folder / f"classification_report_{name}.txt"
    metrics_path = results_folder / f"classification_metrics_{name}.csv"
    predictions_path = results_folder / f"binary_predictions_{name}.npz"
    report_path.write_text(summary, encoding="utf-8")
    pd.DataFrame(report_dict).transpose().to_csv(metrics_path)
    np.savez(predictions_path, data=predictions)

    print(summary)
    print("\nSaved:")
    print(report_path)
    print(metrics_path)
    print(predictions_path)


first_labels, first_scores = load_predictions(first_results_folder)
label_names = validate_label_names(first_labels.shape[1])

if mode == "fixed":
    try:
        threshold = float(mode_value)
    except ValueError as exc:
        raise SystemExit(
            f"THRESHOLD must be a number between 0 and 1: {mode_value}"
        ) from exc
    if not 0.0 <= threshold <= 1.0:
        raise SystemExit(f"THRESHOLD must be between 0 and 1: {threshold}")

    thresholds = np.full(first_labels.shape[1], threshold)
    threshold_tag = f"threshold_{threshold:.6g}".replace(".", "p")
    write_evaluation(
        first_results_folder,
        first_labels,
        first_scores,
        thresholds,
        threshold_tag,
    )
else:
    test_labels, test_scores = load_predictions(second_results_folder)
    if test_labels.shape[1] != first_labels.shape[1]:
        raise SystemExit(
            "Validation and test predictions contain different numbers of labels: "
            f"validation={first_labels.shape[1]}, test={test_labels.shape[1]}"
        )

    thresholds = np.full(first_labels.shape[1], 0.5)
    calibration_rows = []
    for column, label_name in enumerate(label_names):
        validation_labels = first_labels[:, column]
        validation_scores = first_scores[:, column]
        status = "optimized"

        if np.unique(validation_labels).size < 2:
            status = "fallback: validation lacks both classes"
        else:
            precision, recall, candidates = precision_recall_curve(
                validation_labels,
                validation_scores,
            )
            f1 = (
                2 * precision[:-1] * recall[:-1]
                / (precision[:-1] + recall[:-1] + 1e-12)
            )
            best_f1 = np.max(f1)
            best_indices = np.flatnonzero(np.isclose(f1, best_f1))
            thresholds[column] = candidates[best_indices[-1]]

        calibration_rows.append(
            {
                "label": label_name,
                "threshold": thresholds[column],
                "validation_positives": int(validation_labels.sum()),
                "validation_negatives": int(len(validation_labels) - validation_labels.sum()),
                "status": status,
            }
        )

    thresholds_path = first_results_folder / "calibrated_f1_thresholds.npy"
    thresholds_csv_path = first_results_folder / "calibrated_f1_thresholds.csv"
    np.save(thresholds_path, thresholds)
    pd.DataFrame(calibration_rows).to_csv(thresholds_csv_path, index=False)

    print("F1-optimal validation thresholds:")
    print(pd.DataFrame(calibration_rows).to_string(index=False))
    print("\nSaved calibration:")
    print(thresholds_path)
    print(thresholds_csv_path)
    print("\nSeparate test-set evaluation:\n")
    write_evaluation(
        second_results_folder,
        test_labels,
        test_scores,
        thresholds,
        "calibrated_f1",
    )
PY
