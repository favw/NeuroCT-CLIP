import math

import pandas as pd
import torch

CHEST_CT_PATHOLOGIES = [
    "Medical material",
    "Arterial wall calcification",
    "Cardiomegaly",
    "Pericardial effusion",
    "Coronary artery wall calcification",
    "Hiatal hernia",
    "Lymphadenopathy",
    "Emphysema",
    "Atelectasis",
    "Lung nodule",
    "Lung opacity",
    "Pulmonary fibrotic sequela",
    "Pleural effusion",
    "Mosaic attenuation pattern",
    "Peribronchial thickening",
    "Consolidation",
    "Bronchiectasis",
    "Interlobular septal thickening",
]


def load_label_columns(labels_file):
    label_df = pd.read_csv(labels_file, nrows=0)
    if "VolumeName" not in label_df.columns:
        raise KeyError("Labels file must contain a 'VolumeName' column.")
    label_columns = [col for col in label_df.columns if col != "VolumeName"]
    if all(pathology in label_columns for pathology in CHEST_CT_PATHOLOGIES):
        return CHEST_CT_PATHOLOGIES.copy()
    return label_columns


def require_cli_args(args, required_names):
    missing = [name for name in required_names if getattr(args, name) is None]
    if missing:
        raise ValueError(f"Missing required arguments: {', '.join(missing)}")


def require_head_aware_cli_args(args, base_required_names, *, meta_arg_names=("meta_file",)):
    required_names = list(base_required_names)
    if not getattr(args, "head", False):
        required_names.extend(meta_arg_names)
    require_cli_args(args, tuple(required_names))


def _unwrap_state_dict(checkpoint):
    if isinstance(checkpoint, dict) and "model" in checkpoint and isinstance(checkpoint["model"], dict):
        return checkpoint["model"]
    return checkpoint


def infer_checkpoint_channels(checkpoint_path, *, patch_size=20, temporal_patch_size=10):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = _unwrap_state_dict(checkpoint)

    candidate_keys = (
        "visual_transformer.to_patch_emb.2.weight",
        "module.visual_transformer.to_patch_emb.2.weight",
        "trained_model.visual_transformer.to_patch_emb.2.weight",
        "module.trained_model.visual_transformer.to_patch_emb.2.weight",
        "visual_transformer.to_patch_emb.1.weight",
        "module.visual_transformer.to_patch_emb.1.weight",
        "trained_model.visual_transformer.to_patch_emb.1.weight",
        "module.trained_model.visual_transformer.to_patch_emb.1.weight",
    )

    base_features = patch_size * patch_size * temporal_patch_size

    for key in candidate_keys:
        if key not in state_dict:
            continue

        tensor = state_dict[key]
        if tensor.ndim >= 2:
            feature_dim = tensor.shape[-1]
        else:
            feature_dim = tensor.shape[0]

        channels = feature_dim / base_features
        if math.isclose(channels, round(channels)):
            return int(round(channels))

    return None


def assert_head_checkpoint_compatible(args):
    if not args.head or args.pretrained is None:
        return

    checkpoint_channels = infer_checkpoint_channels(args.pretrained)
    if checkpoint_channels == 1:
        raise ValueError(
            "Head-CT mode uses a 4-channel image encoder and cannot directly load a chest 1-channel "
            f"CLIP checkpoint: {args.pretrained}"
        )
