import os
import glob
from functools import partial
from typing import Dict, List, Sequence, Tuple

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import tqdm


def resize_array(array: torch.Tensor, current_spacing: Sequence[float], target_spacing: Sequence[float]) -> np.ndarray:
    """
    Resize a 3D CT tensor to match target voxel spacing.

    Args:
        array: Tensor with shape [1, 1, D, H, W].
        current_spacing: Current spacing ordered as (z, x, y).
        target_spacing: Target spacing ordered as (z, x, y).

    Returns:
        Resampled array as numpy ndarray.
    """
    original_shape = array.shape[2:]
    scaling_factors = [current_spacing[i] / target_spacing[i] for i in range(len(original_shape))]
    new_shape = [max(1, int(original_shape[i] * scaling_factors[i])) for i in range(len(original_shape))]
    resized_array = F.interpolate(array, size=new_shape, mode="trilinear", align_corners=False).cpu().numpy()
    return resized_array


class HeadCTReportDataset(Dataset):
    """
    Head-CT dataset for CT-CLIP style image-text pretraining/fine-tuning.

    Expected inputs:
      - data folder hierarchy containing .nii.gz volumes
      - reports CSV with accession/volume identifier + report text fields
      - metadata CSV with rescale and spacing fields per volume
    """

    def __init__(
        self,
        data_folder: str,
        reports_file: str,
        meta_file: str,
        *,
        # Report schema
        volume_name_col: str = "VolumeName",
        findings_col: str = "Findings_EN",
        impression_col: str = "Impressions_EN",
        # Preprocessing policy for head CT (your convenience defaults)
        target_spacing: Tuple[float, float, float] = (1.5, 0.75, 0.75),  # (z, x, y)
        hu_windows: Sequence[Tuple[int, int]] = ((-100, 200), (-500, 2000), (0, 150), (-1000, -200)),
        target_shape: Tuple[int, int, int] = (512, 512, 256),  # (H, W, D)
        # Dataset filtering / subsampling
        keep_ratio: float = 1.0,
        min_slices: int = 20,
    ):
        self.data_folder = data_folder
        self.reports_file = reports_file
        self.meta_file = meta_file

        self.volume_name_col = volume_name_col
        self.findings_col = findings_col
        self.impression_col = impression_col

        self.target_spacing = target_spacing
        self.hu_windows = [tuple(w) for w in hu_windows]
        self.target_shape = target_shape
        self.keep_ratio = keep_ratio
        self.min_slices = min_slices

        if len(self.hu_windows) == 0:
            raise ValueError("hu_windows must contain at least one HU window.")

        self.accession_to_text = self.load_accession_text(self.reports_file)
        self.paths: List[str] = []
        self.samples = self.prepare_samples()

        if not (0 < self.keep_ratio <= 1.0):
            raise ValueError("keep_ratio must be in (0, 1].")

        if self.keep_ratio < 1.0:
            num_files = max(1, int(len(self.samples) * self.keep_ratio))
            self.samples = self.samples[:num_files]

        self.meta_df = pd.read_csv(self.meta_file)
        self.nii_to_tensor = partial(self.nii_img_to_tensor, df=self.meta_df)

    def _parse_xy_spacing(self, value) -> float:
        """
        Parse XY spacing from scalar or string forms like "(0.6, 0.6)" / "[0.6,0.6]".
        """
        if isinstance(value, (float, int)):
            return float(value)

        text = str(value).strip()
        for ch in "()[]":
            text = text.replace(ch, "")
        parts = [p.strip() for p in text.split(",") if p.strip()]
        if not parts:
            raise ValueError(f"Could not parse XY spacing from value: {value}")
        return float(parts[0])

    def load_accession_text(self, reports_file: str) -> Dict[str, Tuple[str, str]]:
        df = pd.read_csv(reports_file)
        required = [self.volume_name_col, self.findings_col, self.impression_col]
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise KeyError(f"Missing required report columns: {missing}")

        accession_to_text: Dict[str, Tuple[str, str]] = {}
        for _, row in df.iterrows():
            accession_to_text[row[self.volume_name_col]] = (
                row[self.findings_col],
                row[self.impression_col],
            )
        return accession_to_text

    def prepare_samples(self) -> List[Tuple[str, str]]:
        samples: List[Tuple[str, str]] = []
        for patient_folder in tqdm.tqdm(glob.glob(os.path.join(self.data_folder, "*"))):
            for accession_folder in glob.glob(os.path.join(patient_folder, "*")):
                for nii_file in glob.glob(os.path.join(accession_folder, "*.nii.gz")):
                    volume_name = os.path.basename(nii_file)
                    if volume_name not in self.accession_to_text:
                        continue

                    # Simple filter on number of slices before expensive work.
                    try:
                        nii_img = nib.load(nii_file)
                        if nii_img.shape[-1] < self.min_slices:
                            continue
                    except Exception:
                        continue

                    findings, impression = self.accession_to_text[volume_name]
                    findings = "" if str(findings) == "Not given." else str(findings)
                    impression = "" if str(impression) == "Not given." else str(impression)

                    input_text = f"{findings} {impression}".strip()
                    samples.append((nii_file, input_text))
                    self.paths.append(nii_file)

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def nii_img_to_tensor(self, path: str, df: pd.DataFrame) -> torch.Tensor:
        """
        Returns:
            Tensor in shape [C, D, H, W], where C == len(self.hu_windows).
        """
        nii_img = nib.load(str(path))
        img_data = nii_img.get_fdata()

        file_name = os.path.basename(path)
        row = df[df[self.volume_name_col] == file_name]
        if row.empty:
            raise KeyError(f"No metadata row found for volume: {file_name}")

        slope = float(row["RescaleSlope"].iloc[0])
        intercept = float(row["RescaleIntercept"].iloc[0])
        xy_spacing = self._parse_xy_spacing(row["XYSpacing"].iloc[0])
        z_spacing = float(row["ZSpacing"].iloc[0])

        current_spacing = (z_spacing, xy_spacing, xy_spacing)

        # Apply scanner rescale to HU-like space.
        img_data = slope * img_data + intercept

        # Move to D,H,W for interpolation.
        img_data = img_data.transpose(2, 0, 1)
        tensor = torch.tensor(img_data, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        img_data = resize_array(tensor, current_spacing, self.target_spacing)
        img_data = img_data[0][0]  # D,H,W
        img_data = np.transpose(img_data, (1, 2, 0))  # H,W,D

        # Build multi-window channels in H,W,D and stack to C,H,W,D.
        windowed_channels = []
        for hu_min, hu_max in self.hu_windows:
            clipped = np.clip(img_data, hu_min, hu_max)
            denom = max(float(hu_max - hu_min), 1e-6)
            normalized = ((clipped - hu_min) / denom).astype(np.float32)
            windowed_channels.append(normalized)

        tensor = torch.tensor(np.stack(windowed_channels, axis=0), dtype=torch.float32)  # C,H,W,D

        # Center crop/pad to target H,W,D for all channels.
        target_h, target_w, target_d = self.target_shape
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

        # Convert to expected CT-CLIP tensor layout [C, D, H, W].
        tensor = tensor.permute(0, 3, 1, 2)
        return tensor

    def __getitem__(self, index: int):
        nii_file, input_text = self.samples[index]
        video_tensor = self.nii_to_tensor(nii_file)

        input_text = str(input_text)
        input_text = input_text.replace('"', "")
        input_text = input_text.replace("'", "")
        input_text = input_text.replace("(", "")
        input_text = input_text.replace(")", "")

        return video_tensor, input_text
