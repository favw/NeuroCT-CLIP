# NIfTI Downscaler

`nifti-downscaler` is an offline preprocessing package for reducing the voxel
grid of three-dimensional `.nii.gz` volumes. It has no imports from CT-CLIP and
does not modify the training pipeline. Its only boundary with the rest of the
project is the input and output NIfTI files.

## Installation

From the repository root:

```bash
python -m pip install -e ./nifti_downscaler
```

The package uses NiBabel, NumPy, and SciPy. It does not require TorchIO or a GPU.

## Command line

Downscale all three array axes by one half:

```bash
nifti-downscale input.nii.gz output.nii.gz --scale-factor 0.5
```

Keep the in-plane resolution and halve only the axial/depth axis:

```bash
nifti-downscale input.nii.gz output.nii.gz --scale-factor 1 1 0.5
```

Use an explicit output shape in NIfTI array order `(X, Y, Z)`:

```bash
nifti-downscale input.nii.gz output.nii.gz --target-shape 240 240 120
```

Process a dataset recursively while preserving its relative patient/accession
directory structure and file names. The input folder may contain `.nii.gz`
files directly, nested patient folders, or both:

```bash
nifti-downscale \
  --input-folder /data/original \
  --output-folder /data/downscaled \
  --target-shape 240 240 120
```

The older positional form (`nifti-downscale INPUT OUTPUT ...`) remains
available for individual files and existing scripts.

Existing outputs are protected by default. Use `--overwrite` to replace files
under the output path; the tool never permits an input file to overwrite itself.
For a single input file, an output path without a `.nii.gz` suffix is created as
a directory and the original file name is retained inside it.
Run `nifti-downscale --help` for interpolation, anti-aliasing, and recursion
options. Linear interpolation with anti-alias filtering is the default for CT
intensities. `nearest` is available for discrete-valued volumes.

## Python API

```python
from nifti_downscaler import downscale_nifti, downscale_path

result = downscale_nifti(
    "input.nii.gz",
    "output.nii.gz",
    scale_factor=(1.0, 1.0, 0.5),
)
print(result.original_shape, result.output_shape, result.voxel_fraction)

results = downscale_path(
    "/data/original",
    "/data/downscaled",
    target_shape=(240, 240, 120),
)
```

Exactly one of `scale_factor` and `target_shape` is required. This is a
downscaler, so no axis may be enlarged and at least one axis must shrink. Only
three-dimensional `.nii.gz` files are accepted.

## Spatial and intensity behavior

- The output keeps the source orientation, world-space centre, and physical
  field-of-view edges. Voxel sizes and qform/sform transforms are updated for
  the smaller grid.
- NIfTI header slope/intercept scaling is applied before interpolation. Output
  values are stored as float32 with identity scaling, so loading the output with
  `get_fdata()` returns the resampled physical intensities.
- Writes are atomic within the output directory, preventing a failed save from
  leaving a partial destination file.
- Directory processing is sequential so multiple full CT volumes are not held
  in RAM at the same time.

The tool intentionally does not edit external CSV or JSON metadata. If a loader
uses external `XYSpacing`, `ZSpacing`, `RescaleSlope`, or `RescaleIntercept`
instead of the NIfTI header, update that metadata for the downscaled dataset or
configure the loader to trust the output header. Reapplying the old intensity
slope/intercept would scale the data twice.

## CT-CLIP batch-size integration

The current CT-CLIP loaders crop/pad every sample to `(480, 480, 240)`, and the
model fixes the in-plane size to `480 x 480`. Pointing `--data-folder` at smaller
files without changing that input contract will therefore not reduce training
memory: the loader will expand or pad them back to the original tensor shape.

For existing checkpoints, the least disruptive option is usually axial-only
downscaling:

1. Run the tool with `--scale-factor 1 1 0.5`.
2. Configure the loader to use the output header spacing (or update its metadata)
   and a target depth of `120` instead of `240`.
3. Keep height and width at `480`; depth must remain divisible by CTViT's temporal
   patch size of `10`.

Reducing X/Y as well requires matching changes to the CTViT image size and the
CT-CLIP image projection width, so existing projection checkpoint weights will
not be shape-compatible. A factor of `0.5` on every axis retains 12.5% of the
input voxels; axial-only `0.5` retains 50%.

## Tests

```bash
python -m unittest discover -s nifti_downscaler/tests -p 'test_*.py' -v
```
