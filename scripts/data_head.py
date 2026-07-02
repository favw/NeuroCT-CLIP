from typing import List, Sequence, Tuple

import torch
import tqdm
from torch.utils.data import Dataset

from head_volume_utils import (
    build_meta_lookup,
    build_report_lookup,
    compose_report_text,
    discover_head_image_records,
    load_head_tensor,
    resolve_lookup_item,
)


class HeadCTReportDataset(Dataset):
    def __init__(
        self,
        data_folder: str,
        reports_file: str,
        meta_file: str = None,
        *,
        volume_name_col: str = "VolumeName",
        findings_col: str = "Findings_EN",
        impression_col: str = "Impressions_EN",
        target_spacing: Tuple[float, float, float] = (1.5, 0.75, 0.75),
        hu_windows: Sequence[Tuple[int, int]] = ((-100, 200), (-500, 2000), (0, 150), (-1000, -200)),
        target_shape: Tuple[int, int, int] = (480, 480, 240),
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
        self.hu_windows = [tuple(window) for window in hu_windows]
        self.target_shape = target_shape
        self.keep_ratio = keep_ratio
        self.min_slices = min_slices

        if len(self.hu_windows) == 0:
            raise ValueError("hu_windows must contain at least one HU window.")
        if not (0 < self.keep_ratio <= 1.0):
            raise ValueError("keep_ratio must be in (0, 1].")

        self.report_lookup = build_report_lookup(
            self.reports_file,
            volume_name_col=self.volume_name_col,
            findings_col=self.findings_col,
            impression_col=self.impression_col,
        )
        self.meta_lookup = build_meta_lookup(self.meta_file, volume_name_col=self.volume_name_col)

        self.paths: List[str] = []
        self.samples = self.prepare_samples()

        if self.keep_ratio < 1.0:
            num_files = max(1, int(len(self.samples) * self.keep_ratio))
            self.samples = self.samples[:num_files]

    def prepare_samples(self):
        samples = []
        records = discover_head_image_records(self.data_folder, min_slices=self.min_slices)
        missing_report = 0
        example_matches = []

        for record in tqdm.tqdm(records):
            _, report_entry = resolve_lookup_item(self.report_lookup, record.lookup_candidates)
            if report_entry is None:
                missing_report += 1
                continue

            input_text = compose_report_text(report_entry["findings"], report_entry["impression"])
            samples.append((record, input_text))
            self.paths.append(record.image_path)
            if len(example_matches) < 3:
                example_matches.append((record.image_path, report_entry["volume_name"]))

        print(
            f"[head-train] matched {len(samples)}/{len(records)} image records to reports "
            f"({missing_report} without report match)."
        )
        for image_path, volume_name in example_matches:
            print(f"[head-train] example match: {image_path} -> {volume_name}")

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def image_to_tensor(self, record) -> torch.Tensor:
        return load_head_tensor(
            record,
            meta_lookup=self.meta_lookup,
            target_spacing=self.target_spacing,
            hu_windows=self.hu_windows,
            target_shape=self.target_shape,
        )

    def __getitem__(self, index: int):
        record, input_text = self.samples[index]
        video_tensor = self.image_to_tensor(record)

        input_text = str(input_text)
        input_text = input_text.replace('"', "")
        input_text = input_text.replace("'", "")
        input_text = input_text.replace("(", "")
        input_text = input_text.replace(")", "")

        return video_tensor, input_text
