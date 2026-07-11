"""Public API for the standalone NIfTI downscaler."""

from .downscale import DownscaleResult, downscale_nifti, downscale_path

__all__ = ["DownscaleResult", "downscale_nifti", "downscale_path"]
__version__ = "0.1.0"
