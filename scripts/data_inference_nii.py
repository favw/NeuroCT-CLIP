import os
import glob
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from functools import partial
import torch.nn.functional as F
import tqdm
import nibabel as nib

from head_utils import load_label_columns
from head_volume_utils import (
    build_label_lookup,
    build_meta_lookup,
    build_report_lookup,
    compose_report_text,
    discover_head_image_records,
    format_accession_name,
    load_head_tensor,
    resolve_lookup_item,
)

def resize_array(array, current_spacing, target_spacing):
    """
    Resize the array to match the target spacing.

    Args:
    array (torch.Tensor): Input array to be resized.
    current_spacing (tuple): Current voxel spacing (z_spacing, xy_spacing, xy_spacing).
    target_spacing (tuple): Target voxel spacing (target_z_spacing, target_x_spacing, target_y_spacing).

    Returns:
    np.ndarray: Resized array.
    """
    # Calculate new dimensions
    original_shape = array.shape[2:]
    scaling_factors = [
        current_spacing[i] / target_spacing[i] for i in range(len(original_shape))
    ]
    new_shape = [
        int(original_shape[i] * scaling_factors[i]) for i in range(len(original_shape))
    ]
    # Resize the array
    resized_array = F.interpolate(array, size=new_shape, mode='trilinear', align_corners=False).cpu().numpy()
    return resized_array



class CTReportDatasetinfer(Dataset):
    def __init__(self, data_folder, reports_file, meta_file, min_slices=20, labels = "labels.csv"):
        self.data_folder = data_folder
        self.min_slices = min_slices
        self.labels = labels
        self.accession_to_text = self.load_accession_text(reports_file)
        self.paths=[]
        self.samples = self.prepare_samples()
        df = pd.read_csv(meta_file) #select the metadata
        self.nii_to_tensor = partial(self.nii_img_to_tensor, df = df)

    def load_accession_text(self, reports_file):
        df = pd.read_csv(reports_file)
        accession_to_text = {}
        for index, row in df.iterrows():
            accession_to_text[row['VolumeName']] = row["Findings_EN"],row['Impressions_EN']
        return accession_to_text


    def prepare_samples(self):
        samples = []
        patient_folders = glob.glob(os.path.join(self.data_folder, '*'))

        # Read labels once outside the loop
        test_df = pd.read_csv(self.labels)
        test_label_cols = load_label_columns(self.labels)
        test_df['one_hot_labels'] = list(test_df[test_label_cols].values)

        for patient_folder in tqdm.tqdm(patient_folders):
            accession_folders = glob.glob(os.path.join(patient_folder, '*'))

            for accession_folder in accession_folders:
                nii_files = glob.glob(os.path.join(accession_folder, '*.nii.gz'))

                for nii_file in nii_files:
                    accession_number = nii_file.split("/")[-1]

                    if accession_number not in self.accession_to_text:
                        continue

                    impression_text = self.accession_to_text[accession_number]
                    text_final = ""
                    for text in list(impression_text):
                        text = str(text)
                        if text == "Not given.":
                            text = ""

                        text_final = text_final + text

                    onehotlabels = test_df[test_df["VolumeName"] == accession_number]["one_hot_labels"].values
                    if len(onehotlabels) > 0:
                        samples.append((nii_file, text_final, onehotlabels[0]))
                        self.paths.append(nii_file)
        return samples

    def __len__(self):
        return len(self.samples)

    def nii_img_to_tensor(self, path, df):
        nii_img = nib.load(str(path))
        img_data = nii_img.get_fdata()

        file_name = path.split("/")[-1]
        row = df[df['VolumeName'] == file_name]
        slope = float(row["RescaleSlope"].iloc[0])
        intercept = float(row["RescaleIntercept"].iloc[0])
        xy_spacing = float(row["XYSpacing"].iloc[0][1:][:-2].split(",")[0])
        z_spacing = float(row["ZSpacing"].iloc[0])

        # Define the target spacing values
        target_x_spacing = 0.75
        target_y_spacing = 0.75
        target_z_spacing = 1.5

        current = (z_spacing, xy_spacing, xy_spacing)
        target = (target_z_spacing, target_x_spacing, target_y_spacing)

        img_data = slope * img_data + intercept
        hu_min, hu_max = -1000, 1000
        img_data = np.clip(img_data, hu_min, hu_max)

        img_data = img_data.transpose(2, 0, 1)

        tensor = torch.tensor(img_data)
        tensor = tensor.unsqueeze(0).unsqueeze(0)

        img_data = resize_array(tensor, current, target)
        img_data = img_data[0][0]
        img_data= np.transpose(img_data, (1, 2, 0))

        img_data = (((img_data ) / 1000)).astype(np.float32)
        slices=[]

        tensor = torch.tensor(img_data)
        # Get the dimensions of the input tensor
        target_shape = (480,480,240)

        # Extract dimensions
        h, w, d = tensor.shape

        # Calculate cropping/padding values for height, width, and depth
        dh, dw, dd = target_shape
        h_start = max((h - dh) // 2, 0)
        h_end = min(h_start + dh, h)
        w_start = max((w - dw) // 2, 0)
        w_end = min(w_start + dw, w)
        d_start = max((d - dd) // 2, 0)
        d_end = min(d_start + dd, d)

        # Crop or pad the tensor
        tensor = tensor[h_start:h_end, w_start:w_end, d_start:d_end]

        pad_h_before = (dh - tensor.size(0)) // 2
        pad_h_after = dh - tensor.size(0) - pad_h_before

        pad_w_before = (dw - tensor.size(1)) // 2
        pad_w_after = dw - tensor.size(1) - pad_w_before

        pad_d_before = (dd - tensor.size(2)) // 2
        pad_d_after = dd - tensor.size(2) - pad_d_before

        tensor = torch.nn.functional.pad(tensor, (pad_d_before, pad_d_after, pad_w_before, pad_w_after, pad_h_before, pad_h_after), value=-1)

        tensor = tensor.permute(2, 0, 1)

        tensor = tensor.unsqueeze(0)

        return tensor


    def __getitem__(self, index):
        nii_file, input_text, onehotlabels = self.samples[index]
        video_tensor = self.nii_to_tensor(nii_file)
        input_text = input_text.replace('"', '')
        input_text = input_text.replace('\'', '')
        input_text = input_text.replace('(', '')
        input_text = input_text.replace(')', '')
        name_acc = nii_file.split("/")[-1].replace(".nii.gz", "")
        return video_tensor, input_text, onehotlabels, name_acc


class HeadCTReportDatasetinfer(Dataset):
    def __init__(
        self,
        data_folder,
        reports_file,
        meta_file=None,
        min_slices=20,
        labels="labels.csv",
        target_spacing=(1.5, 0.75, 0.75),
        target_shape=(480, 480, 240),
        hu_windows=((-100, 200), (-500, 2000), (0, 150), (-1000, -200)),
        volume_name_col="VolumeName",
        findings_col="Findings_EN",
        impression_col="Impressions_EN",
    ):
        self.data_folder = data_folder
        self.min_slices = min_slices
        self.labels = labels
        self.target_spacing = target_spacing
        self.target_shape = target_shape
        self.hu_windows = [tuple(window) for window in hu_windows]
        self.volume_name_col = volume_name_col
        self.findings_col = findings_col
        self.impression_col = impression_col

        if len(self.hu_windows) == 0:
            raise ValueError("hu_windows must contain at least one HU window.")

        self.report_lookup = build_report_lookup(
            reports_file,
            volume_name_col=self.volume_name_col,
            findings_col=self.findings_col,
            impression_col=self.impression_col,
        )
        self.label_lookup = build_label_lookup(labels, volume_name_col=self.volume_name_col)
        self.meta_lookup = build_meta_lookup(meta_file, volume_name_col=self.volume_name_col)

        self.paths = []
        self.samples = self.prepare_samples()

    def prepare_samples(self):
        samples = []
        records = discover_head_image_records(
            self.data_folder,
            min_slices=self.min_slices,
            allowed_lookup_keys=tuple(self.report_lookup.keys()),
        )
        missing_report = 0
        missing_label = 0

        for record in tqdm.tqdm(records):
            _, report_entry = resolve_lookup_item(self.report_lookup, record.lookup_candidates)
            if report_entry is None:
                missing_report += 1
                continue

            _, label_entry = resolve_lookup_item(self.label_lookup, record.lookup_candidates)
            if label_entry is None:
                missing_label += 1
                continue

            input_text = compose_report_text(report_entry["findings"], report_entry["impression"])
            accession_name = format_accession_name(label_entry["volume_name"])
            samples.append((record, input_text, label_entry["labels"], accession_name))
            self.paths.append(record.image_path)

        print(
            f"[head-valid] discovered={len(records)} matched={len(samples)} "
            f"({missing_report} without report match, {missing_label} without label match)."
        )

        return samples

    def __len__(self):
        return len(self.samples)

    def image_to_tensor(self, record):
        return load_head_tensor(
            record,
            meta_lookup=self.meta_lookup,
            target_spacing=self.target_spacing,
            hu_windows=self.hu_windows,
            target_shape=self.target_shape,
        )

    def __getitem__(self, index):
        record, input_text, onehotlabels, accession_name = self.samples[index]
        video_tensor = self.image_to_tensor(record)
        input_text = input_text.replace('"', "")
        input_text = input_text.replace("'", "")
        input_text = input_text.replace("(", "")
        input_text = input_text.replace(")", "")
        return video_tensor, input_text, onehotlabels, accession_name
