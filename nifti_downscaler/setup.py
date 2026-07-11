import os
from pathlib import Path

from setuptools import find_packages, setup


ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)


setup(
    name="nifti-downscaler",
    version="0.1.0",
    description="Standalone, geometry-preserving downscaling for 3-D NIfTI volumes",
    long_description=(ROOT / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "nibabel>=5.1",
        "numpy>=1.23",
        "scipy>=1.10",
    ],
    entry_points={
        "console_scripts": [
            "nifti-downscale=nifti_downscaler.__main__:main",
        ]
    },
)
