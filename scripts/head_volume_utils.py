import glob
import os
import time
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from logging_utils import timestamped_message

try:
    import pydicom
except ImportError:
    pydicom = None


@dataclass(frozen=True)
class HeadImageRecord:
    storage_type: str
    image_path: str
    lookup_candidates: Tuple[str, ...]
    num_slices: Optional[int] = None


def resize_array(array: torch.Tensor, current_spacing: Sequence[float], target_spacing: Sequence[float]) -> np.ndarray:
    original_shape = array.shape[2:]
    scaling_factors = [current_spacing[i] / target_spacing[i] for i in range(len(original_shape))]
    new_shape = [max(1, int(original_shape[i] * scaling_factors[i])) for i in range(len(original_shape))]
    return F.interpolate(array, size=new_shape, mode="trilinear", align_corners=False).cpu().numpy()


def parse_xy_spacing(value) -> float:
    if isinstance(value, (list, tuple)):
        if not value:
            raise ValueError("Empty XY spacing sequence.")
        return float(value[0])

    if isinstance(value, (float, int)):
        return float(value)

    text = str(value).strip()
    for ch in "()[]":
        text = text.replace(ch, "")
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if not parts:
        raise ValueError(f"Could not parse XY spacing from value: {value}")
    return float(parts[0])


def normalize_volume_key(value) -> Optional[str]:
    if value is None:
        return None

    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None

    text = text.replace("\\", "/").rstrip("/")
    lowered = text.casefold()

    for suffix in (".nii.gz", ".nii", ".npz", ".dcm", ".dicom", ".ima"):
        if lowered.endswith(suffix):
            text = text[: -len(suffix)]
            return text.casefold()

    return lowered


def format_accession_name(value: str) -> str:
    text = str(value).replace("\\", "/").rstrip("/")
    base = os.path.basename(text)
    lowered = base.casefold()

    if lowered.endswith(".nii.gz"):
        return base[:-7]
    if lowered.endswith((".nii", ".npz", ".dcm", ".dicom", ".ima")):
        return os.path.splitext(base)[0]
    return base


def compose_report_text(findings, impression) -> str:
    parts: List[str] = []
    for text in (findings, impression):
        normalized = "" if text is None else str(text).strip()
        if not normalized or normalized == "Not given." or normalized.lower() == "nan":
            continue
        parts.append(normalized)
    return " ".join(parts).strip()


def resolve_lookup_item(lookup: Dict[str, object], candidates: Sequence[str]):
    for key in candidates:
        if key in lookup:
            return key, lookup[key]
    return None, None


def build_report_lookup(
    reports_file: str,
    *,
    volume_name_col: str = "VolumeName",
    findings_col: str = "Findings_EN",
    impression_col: str = "Impressions_EN",
) -> Dict[str, Dict[str, object]]:
    df = pd.read_csv(reports_file)
    required = [volume_name_col, findings_col, impression_col]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required report columns: {missing}")

    lookup: Dict[str, Dict[str, object]] = {}
    for _, row in df.iterrows():
        entry = {
            "volume_name": row[volume_name_col],
            "findings": row[findings_col],
            "impression": row[impression_col],
        }
        for key in _expand_identifier_candidates(row[volume_name_col]):
            lookup.setdefault(key, entry)
    return lookup


def build_label_lookup(labels_file: str, *, volume_name_col: str = "VolumeName") -> Dict[str, Dict[str, object]]:
    df = pd.read_csv(labels_file)
    if volume_name_col not in df.columns:
        raise KeyError(f"Labels file must contain a '{volume_name_col}' column.")

    label_cols = [col for col in df.columns if col != volume_name_col]
    lookup: Dict[str, Dict[str, object]] = {}

    for _, row in df.iterrows():
        entry = {
            "volume_name": row[volume_name_col],
            "labels": np.asarray(row[label_cols].values, dtype=np.float32),
        }
        for key in _expand_identifier_candidates(row[volume_name_col]):
            lookup.setdefault(key, entry)
    return lookup


def build_meta_lookup(meta_file: Optional[str], *, volume_name_col: str = "VolumeName") -> Dict[str, Dict[str, object]]:
    if meta_file is None:
        return {}

    df = pd.read_csv(meta_file)
    if volume_name_col not in df.columns:
        raise KeyError(f"Metadata file must contain a '{volume_name_col}' column.")

    lookup: Dict[str, Dict[str, object]] = {}
    for _, row in df.iterrows():
        entry = row.to_dict()
        for key in _expand_identifier_candidates(row[volume_name_col]):
            lookup.setdefault(key, entry)
    return lookup


def discover_head_image_records(
    data_folder: str,
    *,
    min_slices: int = 20,
    allowed_lookup_keys: Optional[Sequence[str]] = None,
    progress_every: Optional[int] = 1000,
) -> List[HeadImageRecord]:
    records: List[HeadImageRecord] = []
    allowed_lookup_key_set = _normalize_lookup_key_set(allowed_lookup_keys)
    allowed_path_prefixes = _build_allowed_path_prefixes(allowed_lookup_key_set)
    scanned_dirs = 0
    scanned_files = 0

    for root, dirs, files in os.walk(data_folder, topdown=True):
        scanned_dirs += 1
        scanned_files += len(files)
        if progress_every and scanned_dirs % progress_every == 0:
            print(
                timestamped_message(
                    "[head-discover] "
                    f"dirs={scanned_dirs} files={scanned_files} records={len(records)} "
                    f"current={os.path.relpath(root, data_folder)}"
                ),
                flush=True,
            )

        if allowed_path_prefixes:
            dirs[:] = [
                dir_name
                for dir_name in dirs
                if _directory_may_contain_allowed_path(
                    os.path.join(root, dir_name),
                    data_folder,
                    allowed_path_prefixes,
                )
            ]

        visible_files = sorted(file_name for file_name in files if not file_name.startswith("."))
        if not visible_files:
            continue

        nii_files = [
            file_name
            for file_name in visible_files
            if file_name.casefold().endswith(".nii.gz") or file_name.casefold().endswith(".nii")
        ]
        if nii_files:
            for file_name in nii_files:
                image_path = os.path.join(root, file_name)
                lookup_candidates = _build_nifti_candidates(image_path, data_folder)
                if allowed_lookup_key_set and not _any_candidate_in_lookup_set(lookup_candidates, allowed_lookup_key_set):
                    continue

                num_slices = _estimate_nifti_slices(image_path)
                if num_slices is not None and num_slices < min_slices:
                    continue

                records.append(
                    HeadImageRecord(
                        storage_type="nifti",
                        image_path=image_path,
                        lookup_candidates=lookup_candidates,
                        num_slices=num_slices,
                    )
                )
            continue

        if allowed_lookup_key_set and not _dicom_root_may_match_lookup(
            root,
            data_folder,
            visible_files,
            allowed_lookup_key_set,
        ):
            continue

        dicom_files = [os.path.join(root, file_name) for file_name in visible_files if _looks_like_dicom_file(os.path.join(root, file_name))]
        if dicom_files:
            if pydicom is None:
                raise ImportError(
                    "Detected raw DICOM head-CT input under "
                    f"'{root}', but the optional 'pydicom' dependency is not installed."
                )

            num_slices = _estimate_dicom_slices(dicom_files)
            if num_slices is not None and num_slices < min_slices:
                continue

            records.append(
                HeadImageRecord(
                    storage_type="dicom",
                    image_path=root,
                    lookup_candidates=_build_dicom_candidates(root, data_folder, dicom_files),
                    num_slices=num_slices,
                )
            )

    if progress_every:
        print(
            timestamped_message(
                "[head-discover] "
                f"finished dirs={scanned_dirs} files={scanned_files} records={len(records)}"
            ),
            flush=True,
        )

    return sorted(records, key=lambda record: record.image_path)


def discover_head_image_records_from_lookup_values(
    data_folder: str,
    lookup_values: Sequence[object],
    *,
    min_slices: int = 20,
    progress_every: int = 25,
    slow_lookup_seconds: float = 5.0,
) -> List[HeadImageRecord]:
    records_by_path: Dict[Tuple[str, str], HeadImageRecord] = {}
    missing_values: List[object] = []
    seen_lookup_keys: Set[str] = set()
    total_values = len(lookup_values)

    for index, value in enumerate(lookup_values, start=1):
        lookup_candidates = _expand_identifier_candidates(value)
        if not lookup_candidates:
            continue

        primary_key = lookup_candidates[0]
        if primary_key in seen_lookup_keys:
            continue
        seen_lookup_keys.add(primary_key)

        if progress_every and (len(seen_lookup_keys) == 1 or len(seen_lookup_keys) % progress_every == 0):
            print(
                timestamped_message(
                    "[head-discover-targeted] "
                    f"resolving={len(seen_lookup_keys)}/{total_values} "
                    f"records={len(records_by_path)} missing={len(missing_values)} current={value}"
                ),
                flush=True,
            )

        start_time = time.monotonic()
        records = _resolve_head_records_for_lookup_value(
            data_folder,
            value,
            lookup_candidates,
            min_slices=min_slices,
        )
        elapsed = time.monotonic() - start_time
        if elapsed >= slow_lookup_seconds:
            print(
                timestamped_message(
                    "[head-discover-targeted] "
                    f"slow lookup seconds={elapsed:.1f} value={value}"
                ),
                flush=True,
            )
        if not records:
            missing_values.append(value)
            continue

        for record in records:
            records_by_path[(record.storage_type, os.path.abspath(record.image_path))] = record

    print(
        timestamped_message(
            "[head-discover-targeted] "
            f"requested={len(seen_lookup_keys)} records={len(records_by_path)} missing={len(missing_values)}"
        ),
        flush=True,
    )
    if missing_values:
        preview = ", ".join(str(value) for value in missing_values[:5])
        suffix = "..." if len(missing_values) > 5 else ""
        print(timestamped_message(f"[head-discover-targeted] missing examples: {preview}{suffix}"), flush=True)

    return sorted(records_by_path.values(), key=lambda record: record.image_path)


def load_head_tensor(
    record: HeadImageRecord,
    *,
    meta_lookup: Optional[Dict[str, Dict[str, object]]] = None,
    target_spacing: Tuple[float, float, float],
    hu_windows: Sequence[Tuple[int, int]],
    target_shape: Tuple[int, int, int],
) -> torch.Tensor:
    img_data, current_spacing = load_head_volume(record, meta_lookup=meta_lookup)

    img_data = img_data.transpose(2, 0, 1)
    tensor = torch.tensor(img_data, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    img_data = resize_array(tensor, current_spacing, target_spacing)
    img_data = img_data[0][0]
    img_data = np.transpose(img_data, (1, 2, 0))

    windowed_channels = []
    for hu_min, hu_max in hu_windows:
        clipped = np.clip(img_data, hu_min, hu_max)
        denom = max(float(hu_max - hu_min), 1e-6)
        normalized = ((clipped - hu_min) / denom).astype(np.float32)
        windowed_channels.append(normalized)

    tensor = torch.tensor(np.stack(windowed_channels, axis=0), dtype=torch.float32)
    target_h, target_w, target_d = target_shape

    _, h, w, d = tensor.shape
    h_start = max((h - target_h) // 2, 0)
    h_end = min(h_start + target_h, h)
    w_start = max((w - target_w) // 2, 0)
    w_end = min(w_start + target_w, w)
    d_start = max((d - target_d) // 2, 0)
    d_end = min(d_start + target_d, d)

    tensor = tensor[:, h_start:h_end, w_start:w_end, d_start:d_end]

    pad_h_before = (target_h - tensor.size(1)) // 2
    pad_h_after = target_h - tensor.size(1) - pad_h_before
    pad_w_before = (target_w - tensor.size(2)) // 2
    pad_w_after = target_w - tensor.size(2) - pad_w_before
    pad_d_before = (target_d - tensor.size(3)) // 2
    pad_d_after = target_d - tensor.size(3) - pad_d_before

    tensor = F.pad(
        tensor,
        (pad_d_before, pad_d_after, pad_w_before, pad_w_after, pad_h_before, pad_h_after),
        value=-1,
    )

    return tensor.permute(0, 3, 1, 2)


def load_head_volume(
    record: HeadImageRecord,
    *,
    meta_lookup: Optional[Dict[str, Dict[str, object]]] = None,
) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    meta_lookup = meta_lookup or {}

    if record.storage_type == "dicom":
        return _load_dicom_series(record.image_path)
    if record.storage_type != "nifti":
        raise ValueError(f"Unsupported head image storage type: {record.storage_type}")

    nii_img = nib.load(str(record.image_path))
    img_data = nii_img.get_fdata()
    meta_row = resolve_lookup_item(meta_lookup, record.lookup_candidates)[1]

    zooms = nii_img.header.get_zooms()
    fallback_xy_spacing = float(zooms[0]) if len(zooms) > 0 else 1.0
    fallback_z_spacing = float(zooms[2]) if len(zooms) > 2 else 1.0

    slope = 1.0
    intercept = 0.0
    xy_spacing = fallback_xy_spacing
    z_spacing = fallback_z_spacing

    if meta_row is not None:
        slope = _safe_float(meta_row.get("RescaleSlope"), 1.0)
        intercept = _safe_float(meta_row.get("RescaleIntercept"), 0.0)
        if meta_row.get("XYSpacing") is not None and str(meta_row.get("XYSpacing")).strip().lower() != "nan":
            xy_spacing = parse_xy_spacing(meta_row.get("XYSpacing"))
        z_spacing = _safe_float(meta_row.get("ZSpacing"), fallback_z_spacing)

    img_data = slope * img_data + intercept
    return img_data, (z_spacing, xy_spacing, xy_spacing)


def _expand_identifier_candidates(value) -> Tuple[str, ...]:
    if value is None:
        return tuple()

    text = str(value).strip().replace("\\", "/").rstrip("/")
    if not text:
        return tuple()

    return _dedupe_candidates([text, os.path.basename(text)])


def _dedupe_candidates(candidates: Sequence[object]) -> Tuple[str, ...]:
    normalized: List[str] = []
    seen = set()

    for candidate in candidates:
        for expanded_candidate in _candidate_lookup_variants(candidate):
            key = normalize_volume_key(expanded_candidate)
            if key is None or key in seen:
                continue
            normalized.append(key)
            seen.add(key)

    return tuple(normalized)


def _candidate_lookup_variants(candidate: object) -> Tuple[object, ...]:
    key = normalize_volume_key(candidate)
    if key is None:
        return tuple()

    variants: List[object] = [candidate]
    base = os.path.basename(key)
    if base and base != key:
        variants.append(base)

    for value in (key, base):
        suffix = _underscore_id_suffix(value)
        if suffix is not None:
            variants.append(suffix)

    return tuple(variants)


def _underscore_id_suffix(value: str) -> Optional[str]:
    if "_" not in value:
        return None

    suffix = value.rsplit("_", 1)[-1].strip()
    if not suffix or suffix == value:
        return None

    # Avoid turning ordinary underscore-separated words into lookup keys.
    if not any(char.isdigit() for char in suffix):
        return None

    return suffix


def _normalize_lookup_key_set(values: Optional[Sequence[str]]) -> Optional[Set[str]]:
    if values is None:
        return None

    normalized = {key for key in (normalize_volume_key(value) for value in values) if key is not None}
    return normalized or None


def _build_allowed_path_prefixes(allowed_lookup_keys: Optional[Set[str]]) -> Set[str]:
    if not allowed_lookup_keys:
        return set()

    prefixes: Set[str] = set()
    for key in allowed_lookup_keys:
        if "/" not in key:
            continue

        parts = [part for part in key.split("/") if part]
        for index in range(1, len(parts) + 1):
            prefixes.add("/".join(parts[:index]))

    return prefixes


def _directory_may_contain_allowed_path(directory_path: str, data_folder: str, allowed_path_prefixes: Set[str]) -> bool:
    directory_key = normalize_volume_key(os.path.relpath(directory_path, data_folder))
    if directory_key is None:
        return False
    return directory_key in allowed_path_prefixes


def _any_candidate_in_lookup_set(candidates: Sequence[str], allowed_lookup_keys: Set[str]) -> bool:
    return any(candidate in allowed_lookup_keys for candidate in candidates)


def _build_nifti_candidates(image_path: str, data_folder: str) -> Tuple[str, ...]:
    rel_path = os.path.relpath(image_path, data_folder)
    return _dedupe_candidates([rel_path, os.path.basename(image_path), os.path.basename(os.path.dirname(image_path))])


def _build_dicom_candidates(series_dir: str, data_folder: str, dicom_files: Sequence[str]) -> Tuple[str, ...]:
    candidates: List[object] = [os.path.relpath(series_dir, data_folder), os.path.basename(series_dir)]

    if dicom_files:
        candidates.extend(_read_dicom_lookup_candidates(dicom_files[0]))

    return _dedupe_candidates(candidates)


def _resolve_head_records_for_lookup_value(
    data_folder: str,
    value: object,
    lookup_candidates: Sequence[str],
    *,
    min_slices: int,
) -> List[HeadImageRecord]:
    allowed_lookup_keys = set(lookup_candidates)
    records: List[HeadImageRecord] = []

    for path in _candidate_paths_for_lookup_value(data_folder, value):
        if os.path.isfile(path):
            record = _record_from_candidate_file(path, data_folder, min_slices=min_slices)
            if record is not None and _any_candidate_in_lookup_set(record.lookup_candidates, allowed_lookup_keys):
                records.append(record)
            continue

        if os.path.isdir(path):
            records.extend(
                record
                for record in _records_from_candidate_dir(path, data_folder, min_slices=min_slices)
                if _any_candidate_in_lookup_set(record.lookup_candidates, allowed_lookup_keys)
            )

    return records


def _candidate_paths_for_lookup_value(data_folder: str, value: object) -> Tuple[str, ...]:
    text = str(value).strip().replace("\\", "/").rstrip("/")
    if not text or text.lower() == "nan":
        return tuple()

    candidates: List[str] = []
    raw_values = [text]
    stripped_text = _strip_supported_extension(text)
    if stripped_text != text:
        raw_values.append(stripped_text)
    normalized = normalize_volume_key(text)
    if normalized is not None and normalized != text.casefold():
        raw_values.append(normalized)

    base = os.path.basename(text)
    if base and base not in raw_values:
        raw_values.append(base)
    stripped_base = _strip_supported_extension(base)
    if stripped_base and stripped_base not in raw_values:
        raw_values.append(stripped_base)

    for raw_value in raw_values:
        possible_paths = [raw_value] if os.path.isabs(raw_value) else [os.path.join(data_folder, raw_value)]
        if "/" in raw_value:
            possible_paths.append(os.path.join(data_folder, os.path.basename(raw_value)))

        for path in possible_paths:
            candidates.append(path)
            lowered = path.casefold()
            if not lowered.endswith((".nii.gz", ".nii", ".dcm", ".dicom", ".ima")):
                candidates.extend([f"{path}.nii.gz", f"{path}.nii"])

        if not os.path.isabs(raw_value):
            candidates.extend(_bounded_lookup_glob_paths(data_folder, raw_value))

    seen: Set[str] = set()
    existing_paths: List[str] = []
    for path in candidates:
        normalized_path = os.path.abspath(path)
        if normalized_path in seen or not os.path.exists(path):
            continue
        seen.add(normalized_path)
        existing_paths.append(path)

    return tuple(existing_paths)


def _strip_supported_extension(value: str) -> str:
    lowered = value.casefold()
    for suffix in (".nii.gz", ".nii", ".npz", ".dcm", ".dicom", ".ima"):
        if lowered.endswith(suffix):
            return value[: -len(suffix)]
    return value


def _bounded_lookup_glob_paths(data_folder: str, value: str) -> List[str]:
    if "/" in value:
        return []

    paths: List[str] = []
    for prefix in ("*", "*/*"):
        base_pattern = os.path.join(data_folder, prefix, value)
        paths.append(base_pattern)
        lowered = value.casefold()
        if not lowered.endswith((".nii.gz", ".nii", ".dcm", ".dicom", ".ima")):
            paths.extend([f"{base_pattern}.nii.gz", f"{base_pattern}.nii"])

    matches: List[str] = []
    for path_pattern in paths:
        matches.extend(glob.glob(path_pattern))
    return matches


def _record_from_candidate_file(
    image_path: str,
    data_folder: str,
    *,
    min_slices: int,
) -> Optional[HeadImageRecord]:
    lowered = image_path.casefold()
    if lowered.endswith((".nii.gz", ".nii")):
        num_slices = _estimate_nifti_slices(image_path)
        if num_slices is not None and num_slices < min_slices:
            return None
        return HeadImageRecord(
            storage_type="nifti",
            image_path=image_path,
            lookup_candidates=_build_nifti_candidates(image_path, data_folder),
            num_slices=num_slices,
        )

    if _looks_like_dicom_file(image_path):
        return _record_from_candidate_dicom_dir(os.path.dirname(image_path), data_folder, min_slices=min_slices)

    return None


def _records_from_candidate_dir(
    directory_path: str,
    data_folder: str,
    *,
    min_slices: int,
) -> List[HeadImageRecord]:
    try:
        visible_files = sorted(file_name for file_name in os.listdir(directory_path) if not file_name.startswith("."))
    except OSError:
        return []

    records: List[HeadImageRecord] = []
    nii_files = [
        file_name
        for file_name in visible_files
        if file_name.casefold().endswith(".nii.gz") or file_name.casefold().endswith(".nii")
    ]
    for file_name in nii_files:
        record = _record_from_candidate_file(
            os.path.join(directory_path, file_name),
            data_folder,
            min_slices=min_slices,
        )
        if record is not None:
            records.append(record)

    if records:
        return records

    dicom_files = [os.path.join(directory_path, file_name) for file_name in visible_files if _looks_like_dicom_file(os.path.join(directory_path, file_name))]
    if dicom_files:
        record = _record_from_candidate_dicom_dir(directory_path, data_folder, min_slices=min_slices)
        return [] if record is None else [record]

    return []


def _record_from_candidate_dicom_dir(
    directory_path: str,
    data_folder: str,
    *,
    min_slices: int,
) -> Optional[HeadImageRecord]:
    try:
        visible_files = sorted(file_name for file_name in os.listdir(directory_path) if not file_name.startswith("."))
    except OSError:
        return None

    dicom_files = [os.path.join(directory_path, file_name) for file_name in visible_files if _looks_like_dicom_file(os.path.join(directory_path, file_name))]
    if not dicom_files:
        return None
    if pydicom is None:
        raise ImportError(
            "Detected raw DICOM head-CT input under "
            f"'{directory_path}', but the optional 'pydicom' dependency is not installed."
        )

    num_slices = _estimate_dicom_slices(dicom_files)
    if num_slices is not None and num_slices < min_slices:
        return None

    return HeadImageRecord(
        storage_type="dicom",
        image_path=directory_path,
        lookup_candidates=_build_dicom_candidates(directory_path, data_folder, dicom_files),
        num_slices=num_slices,
    )


def _dicom_root_may_match_lookup(
    root: str,
    data_folder: str,
    visible_files: Sequence[str],
    allowed_lookup_keys: Set[str],
    *,
    max_probe_files: int = 3,
) -> bool:
    root_candidates = _dedupe_candidates([os.path.relpath(root, data_folder), os.path.basename(root)])
    if _any_candidate_in_lookup_set(root_candidates, allowed_lookup_keys):
        return True

    probe_paths = _choose_dicom_probe_paths(root, visible_files, max_probe_files=max_probe_files)
    for file_path in probe_paths:
        if _any_candidate_in_lookup_set(_read_dicom_lookup_candidates(file_path), allowed_lookup_keys):
            return True

    return False


def _choose_dicom_probe_paths(root: str, visible_files: Sequence[str], *, max_probe_files: int) -> List[str]:
    explicit_dicom = [
        os.path.join(root, file_name)
        for file_name in visible_files
        if os.path.splitext(file_name)[1].casefold() in {".dcm", ".dicom", ".ima"}
    ]
    if explicit_dicom:
        return explicit_dicom[:max_probe_files]

    return [os.path.join(root, file_name) for file_name in visible_files[:max_probe_files]]


def _read_dicom_lookup_candidates(file_path: str) -> Tuple[str, ...]:
    if pydicom is None:
        return tuple()

    try:
        ds = pydicom.dcmread(file_path, stop_before_pixels=True, force=True)
    except Exception:
        return tuple()

    return _dedupe_candidates(
        getattr(ds, attr, None)
        for attr in ("AccessionNumber", "SeriesInstanceUID", "StudyInstanceUID", "SeriesDescription")
    )


def _estimate_nifti_slices(image_path: str) -> Optional[int]:
    try:
        nii_img = nib.load(str(image_path))
    except Exception:
        return None

    if len(nii_img.shape) < 3:
        return None
    return int(nii_img.shape[-1])


def _estimate_dicom_slices(dicom_files: Sequence[str]) -> Optional[int]:
    if len(dicom_files) != 1:
        return len(dicom_files)

    if pydicom is None:
        return len(dicom_files)

    try:
        ds = pydicom.dcmread(dicom_files[0], stop_before_pixels=True, force=True)
        return int(getattr(ds, "NumberOfFrames", 1) or 1)
    except Exception:
        return len(dicom_files)


def _looks_like_dicom_file(path: str) -> bool:
    ext = os.path.splitext(path)[1].casefold()
    if ext in {".dcm", ".dicom", ".ima"}:
        return True

    try:
        with open(path, "rb") as handle:
            header = handle.read(132)
    except OSError:
        return False

    if len(header) >= 132 and header[128:132] == b"DICM":
        return True

    if pydicom is None:
        return False

    try:
        pydicom.dcmread(path, stop_before_pixels=True, force=True)
        return True
    except Exception:
        return False


def _load_dicom_series(series_dir: str) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    if pydicom is None:
        raise ImportError(
            "Raw DICOM head-CT input requires the optional 'pydicom' dependency. "
            "Install pydicom before using a DICOM series directory."
        )

    dicom_files = sorted(
        os.path.join(series_dir, file_name)
        for file_name in os.listdir(series_dir)
        if _looks_like_dicom_file(os.path.join(series_dir, file_name))
    )
    if not dicom_files:
        raise FileNotFoundError(f"No DICOM slices found in series directory: {series_dir}")

    slices = []
    for file_path in dicom_files:
        ds = pydicom.dcmread(file_path, force=True)
        if not hasattr(ds, "PixelData"):
            continue
        slices.append((file_path, ds))

    if not slices:
        raise ValueError(f"Could not read any pixel-bearing DICOM slices from: {series_dir}")

    if len(slices) == 1 and int(getattr(slices[0][1], "NumberOfFrames", 1) or 1) > 1:
        ds = slices[0][1]
        volume = ds.pixel_array.astype(np.float32)
        if volume.ndim == 2:
            volume = volume[np.newaxis, ...]
        slope = _safe_float(getattr(ds, "RescaleSlope", 1.0), 1.0)
        intercept = _safe_float(getattr(ds, "RescaleIntercept", 0.0), 0.0)
        volume = volume * slope + intercept
        volume = np.transpose(volume, (1, 2, 0))
        xy_spacing = parse_xy_spacing(getattr(ds, "PixelSpacing", [1.0, 1.0]))
        z_spacing = _safe_float(
            getattr(ds, "SpacingBetweenSlices", getattr(ds, "SliceThickness", 1.0)),
            1.0,
        )
        return volume, (z_spacing, xy_spacing, xy_spacing)

    ordered = []
    for fallback_index, (file_path, ds) in enumerate(slices):
        z_position = _extract_z_position(ds)
        instance_number = _safe_float(getattr(ds, "InstanceNumber", fallback_index), float(fallback_index))
        ordered.append((z_position, instance_number, file_path, ds))

    ordered.sort(key=lambda item: (item[0] is None, item[0] if item[0] is not None else item[1], item[1], item[2]))

    prepared_slices = []
    shape_counts = Counter()
    z_positions: List[float] = []

    for z_position, _, file_path, ds in ordered:
        try:
            pixels = ds.pixel_array.astype(np.float32)
        except Exception as exc:
            raise ValueError(f"Could not decode DICOM pixel data in '{file_path}'.") from exc

        shape = tuple(pixels.shape)
        shape_counts[shape] += 1
        prepared_slices.append((z_position, file_path, pixels, ds))

    if len(shape_counts) > 1:
        target_shape, keep_count = shape_counts.most_common(1)[0]
        skipped_count = len(prepared_slices) - keep_count
        print(
            timestamped_message(
                "[head-dicom] "
                f"skipping {skipped_count} slice(s) with non-dominant shapes in {series_dir}; "
                f"using shape={target_shape} count={keep_count}"
            ),
            flush=True,
        )
        prepared_slices = [
            item for item in prepared_slices
            if tuple(item[2].shape) == target_shape
        ]

    if not prepared_slices:
        raise ValueError(f"No consistently shaped DICOM slices found in series directory: {series_dir}")

    slice_arrays = []
    for z_position, _, pixels, ds in prepared_slices:
        slope = _safe_float(getattr(ds, "RescaleSlope", 1.0), 1.0)
        intercept = _safe_float(getattr(ds, "RescaleIntercept", 0.0), 0.0)
        slice_arrays.append(pixels * slope + intercept)
        if z_position is not None:
            z_positions.append(z_position)

    volume = np.stack(slice_arrays, axis=-1)
    first_ds = prepared_slices[0][3]
    xy_spacing = parse_xy_spacing(getattr(first_ds, "PixelSpacing", [1.0, 1.0]))
    z_spacing = _infer_dicom_z_spacing(first_ds, z_positions)

    return volume, (z_spacing, xy_spacing, xy_spacing)


def _extract_z_position(ds) -> Optional[float]:
    image_position = getattr(ds, "ImagePositionPatient", None)
    if image_position is not None and len(image_position) >= 3:
        try:
            return float(image_position[2])
        except (TypeError, ValueError):
            pass

    for attr in ("SliceLocation", "InstanceNumber"):
        value = getattr(ds, attr, None)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue

    return None


def _infer_dicom_z_spacing(ds, z_positions: Sequence[float]) -> float:
    if len(z_positions) >= 2:
        diffs = np.diff(sorted(z_positions))
        diffs = np.abs(diffs[diffs != 0])
        if diffs.size > 0:
            return float(np.median(diffs))

    return _safe_float(
        getattr(ds, "SpacingBetweenSlices", getattr(ds, "SliceThickness", 1.0)),
        1.0,
    )


def _safe_float(value, default: float) -> float:
    if value is None:
        return default

    text = str(value).strip()
    if not text or text.lower() == "nan":
        return default

    try:
        return float(value)
    except (TypeError, ValueError):
        return default
