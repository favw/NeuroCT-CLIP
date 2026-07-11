"""Geometry-preserving downscaling for three-dimensional NIfTI images."""

from __future__ import annotations

import math
import os
import tempfile
import zlib
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple, Union

import nibabel as nib
import numpy as np
from nibabel.affines import voxel_sizes
from nibabel.filebasedimages import ImageFileError
from nibabel.processing import resample_from_to
from scipy.ndimage import gaussian_filter


PathLike = Union[str, os.PathLike]
Shape3D = Tuple[int, int, int]
ScaleFactor = Union[float, Sequence[float]]

_INTERPOLATION_ORDERS = {
    "nearest": 0,
    "linear": 1,
    "cubic": 3,
}


@dataclass(frozen=True)
class DownscaleResult:
    """Summary of one successfully downscaled volume."""

    input_path: Path
    output_path: Path
    original_shape: Shape3D
    output_shape: Shape3D
    original_zooms: Tuple[float, float, float]
    output_zooms: Tuple[float, float, float]

    @property
    def voxel_fraction(self) -> float:
        """Fraction of source voxels retained in the output grid."""

        return math.prod(self.output_shape) / math.prod(self.original_shape)


def downscale_nifti(
    input_path: PathLike,
    output_path: PathLike,
    *,
    scale_factor: Optional[ScaleFactor] = None,
    target_shape: Optional[Sequence[int]] = None,
    interpolation: str = "linear",
    antialias: bool = True,
    overwrite: bool = False,
) -> DownscaleResult:
    """Downscale one ``.nii.gz`` image and write it to a separate path.

    Exactly one of ``scale_factor`` and ``target_shape`` must be supplied.
    A scalar factor applies to all three array axes; a three-value factor is in
    NIfTI array order (X, Y, Z). Factors may equal one for axes that should stay
    unchanged, but at least one axis must be reduced.

    The output affine is adjusted so that the original physical field of view,
    orientation, and volume centre are retained. Intensity scaling stored in the
    input header is applied before interpolation, and the result is stored as
    float32 with identity intensity scaling.
    """

    source = Path(input_path).expanduser()
    destination = Path(output_path).expanduser()

    _validate_input_file(source)
    _validate_output_file(source, destination, overwrite=overwrite)

    if (scale_factor is None) == (target_shape is None):
        raise ValueError("Provide exactly one of scale_factor or target_shape.")

    interpolation_key = str(interpolation).casefold()
    if interpolation_key not in _INTERPOLATION_ORDERS:
        choices = ", ".join(sorted(_INTERPOLATION_ORDERS))
        raise ValueError(f"interpolation must be one of: {choices}.")

    try:
        image = nib.load(str(source))
    except ImageFileError as exc:
        raise ValueError(f"Could not read NIfTI image {source}: {exc}") from exc
    if len(image.shape) != 3:
        raise ValueError(
            f"Only 3-D NIfTI images are supported; {source} has shape {image.shape}."
        )

    original_shape = tuple(int(size) for size in image.shape)
    if any(size < 1 for size in original_shape):
        raise ValueError(f"Input image has an invalid shape: {original_shape}.")

    if scale_factor is not None:
        factors = _normalise_scale_factor(scale_factor)
        output_shape = tuple(
            max(1, int(math.floor(size * factor + 0.5)))
            for size, factor in zip(original_shape, factors)
        )
        if output_shape == original_shape:
            raise ValueError(
                "The requested scale_factor does not reduce this image at its current size."
            )
    else:
        output_shape = _normalise_target_shape(target_shape)
        if any(new > old for new, old in zip(output_shape, original_shape)):
            raise ValueError(
                f"target_shape {output_shape} would enlarge input shape {original_shape}."
            )
        if output_shape == original_shape:
            raise ValueError("target_shape must reduce at least one image axis.")

    source_affine = np.asarray(image.affine, dtype=np.float64)
    if source_affine.shape != (4, 4) or not np.all(np.isfinite(source_affine)):
        raise ValueError("Input image does not contain a finite 4x4 affine.")
    if np.linalg.matrix_rank(source_affine[:3, :3]) < 3:
        raise ValueError("Input image affine is singular and cannot be resampled.")

    target_affine = _scaled_affine(source_affine, original_shape, output_shape)
    try:
        output_data = _resample_data(
            image,
            output_shape,
            target_affine,
            interpolation_order=_INTERPOLATION_ORDERS[interpolation_key],
            antialias=bool(antialias),
        )
    except (ImageFileError, EOFError, OSError, zlib.error) as exc:
        raise ValueError(f"Could not read NIfTI voxel data {source}: {exc}") from exc
    output_image = _build_output_image(
        image,
        output_data,
        target_affine,
        original_shape,
        output_shape,
    )

    destination.parent.mkdir(parents=True, exist_ok=True)
    if overwrite and destination.exists():
        file_mode = destination.stat().st_mode & 0o777
    else:
        file_mode = source.stat().st_mode & 0o777
    _atomic_save(
        output_image,
        destination,
        file_mode=file_mode,
        overwrite=overwrite,
    )

    return DownscaleResult(
        input_path=source,
        output_path=destination,
        original_shape=original_shape,
        output_shape=output_shape,
        original_zooms=tuple(float(value) for value in voxel_sizes(source_affine)),
        output_zooms=tuple(float(value) for value in voxel_sizes(target_affine)),
    )


def downscale_path(
    input_path: PathLike,
    output_path: PathLike,
    *,
    scale_factor: Optional[ScaleFactor] = None,
    target_shape: Optional[Sequence[int]] = None,
    interpolation: str = "linear",
    antialias: bool = True,
    overwrite: bool = False,
    recursive: bool = True,
) -> list[DownscaleResult]:
    """Downscale one file or a directory tree of ``.nii.gz`` files.

    Directory inputs are mirrored below ``output_path`` so patient/accession
    layout and file names remain unchanged. Processing is deliberately
    sequential to avoid holding multiple medical volumes in memory at once.
    """

    source = Path(input_path).expanduser()
    destination = Path(output_path).expanduser()

    if source.is_file():
        output_is_directory = destination.is_dir() or (
            not destination.exists() and not _has_nii_gz_suffix(destination)
        )
        output_file = destination / source.name if output_is_directory else destination
        return [
            downscale_nifti(
                source,
                output_file,
                scale_factor=scale_factor,
                target_shape=target_shape,
                interpolation=interpolation,
                antialias=antialias,
                overwrite=overwrite,
            )
        ]

    if not source.exists():
        raise FileNotFoundError(f"Input path does not exist: {source}")
    if not source.is_dir():
        raise ValueError(f"Input path is neither a file nor a directory: {source}")
    if destination.exists() and not destination.is_dir():
        raise NotADirectoryError(
            f"A directory input requires a directory output: {destination}"
        )

    source_resolved = source.resolve()
    destination_resolved = destination.resolve()
    if source_resolved == destination_resolved:
        raise ValueError("Input and output directories must be different.")

    candidates: Iterable[Path] = source.rglob("*") if recursive else source.iterdir()
    output_is_nested = _is_relative_to(destination_resolved, source_resolved)
    input_files = []
    for candidate in candidates:
        if not candidate.is_file() or not _has_nii_gz_suffix(candidate):
            continue
        if output_is_nested and _is_relative_to(candidate.resolve(), destination_resolved):
            continue
        input_files.append(candidate)

    input_files.sort(key=lambda path: path.as_posix().casefold())
    if not input_files:
        scope = "recursively" if recursive else "at the top level"
        raise FileNotFoundError(f"No .nii.gz files found {scope} in {source}.")

    results = []
    for input_file in input_files:
        relative_path = input_file.relative_to(source)
        results.append(
            downscale_nifti(
                input_file,
                destination / relative_path,
                scale_factor=scale_factor,
                target_shape=target_shape,
                interpolation=interpolation,
                antialias=antialias,
                overwrite=overwrite,
            )
        )
    return results


def _normalise_scale_factor(scale_factor: ScaleFactor) -> Tuple[float, float, float]:
    if isinstance(scale_factor, Real) and not isinstance(scale_factor, bool):
        values = (float(scale_factor),) * 3
    else:
        try:
            raw_values = tuple(scale_factor)
        except (TypeError, ValueError) as exc:
            raise ValueError("scale_factor must contain one or three numbers.") from exc
        if any(isinstance(value, bool) for value in raw_values):
            raise ValueError("scale_factor must contain one or three numbers.")
        try:
            values = tuple(float(value) for value in raw_values)
        except (TypeError, ValueError) as exc:
            raise ValueError("scale_factor must contain one or three numbers.") from exc
        if len(values) == 1:
            values = values * 3
        if len(values) != 3:
            raise ValueError("scale_factor must contain one or three numbers.")

    if any(not math.isfinite(value) or value <= 0 or value > 1 for value in values):
        raise ValueError("Each scale factor must be finite and in the interval (0, 1].")
    if all(value == 1 for value in values):
        raise ValueError("At least one scale factor must be less than 1.")
    return values


def _normalise_target_shape(target_shape: Optional[Sequence[int]]) -> Shape3D:
    if target_shape is None:
        raise ValueError("target_shape is required.")
    try:
        values = tuple(target_shape)
    except TypeError as exc:
        raise ValueError("target_shape must contain exactly three integers.") from exc
    if len(values) != 3:
        raise ValueError("target_shape must contain exactly three integers.")
    if any(isinstance(value, bool) or not isinstance(value, Integral) for value in values):
        raise ValueError("target_shape must contain exactly three integers.")
    shape = tuple(int(value) for value in values)
    if any(value < 1 for value in shape):
        raise ValueError("Every target_shape dimension must be at least 1.")
    return shape


def _scaled_affine(
    affine: np.ndarray,
    original_shape: Sequence[int],
    output_shape: Sequence[int],
) -> np.ndarray:
    """Scale voxel axes while preserving orientation, centre, and FOV edges."""

    original = np.asarray(original_shape, dtype=np.float64)
    output = np.asarray(output_shape, dtype=np.float64)
    axis_scale = original / output

    scaled = np.eye(4, dtype=np.float64)
    scaled[:3, :3] = affine[:3, :3] @ np.diag(axis_scale)

    original_centre = (original - 1.0) / 2.0
    output_centre = (output - 1.0) / 2.0
    centre_world = affine[:3, :3] @ original_centre + affine[:3, 3]
    scaled[:3, 3] = centre_world - scaled[:3, :3] @ output_centre
    return scaled


def _resample_data(
    image,
    output_shape: Shape3D,
    target_affine: np.ndarray,
    *,
    interpolation_order: int,
    antialias: bool,
) -> np.ndarray:
    data = image.get_fdata(dtype=np.float32)

    if antialias and interpolation_order > 0:
        reduction = np.asarray(image.shape, dtype=np.float64) / np.asarray(
            output_shape, dtype=np.float64
        )
        sigma = np.maximum((reduction - 1.0) / 2.0, 0.0)
        if np.any(sigma > 0):
            data = gaussian_filter(data, sigma=sigma, mode="nearest")

    working_header = image.header.copy()
    working_header.set_data_dtype(np.float32)
    working_image = image.__class__(
        data,
        image.affine,
        header=working_header,
        extra=dict(image.extra),
    )
    working_image.header.set_slope_inter(1.0, 0.0)

    resampled = resample_from_to(
        working_image,
        (output_shape, target_affine),
        order=interpolation_order,
        mode="nearest",
        out_class=image.__class__,
    )
    return np.asarray(resampled.dataobj, dtype=np.float32)


def _build_output_image(
    source_image,
    output_data: np.ndarray,
    target_affine: np.ndarray,
    original_shape: Shape3D,
    output_shape: Shape3D,
):
    output_header = source_image.header.copy()
    output_header.set_data_dtype(np.float32)
    output_image = source_image.__class__(
        output_data,
        target_affine,
        header=output_header,
        extra=dict(source_image.extra),
    )
    output_image.header.set_slope_inter(1.0, 0.0)

    qform, qform_code = source_image.get_qform(coded=True)
    sform, sform_code = source_image.get_sform(coded=True)
    qform_code = int(qform_code)
    sform_code = int(sform_code)

    if qform_code > 0 and qform is not None:
        output_image.set_qform(
            _scaled_affine(qform, original_shape, output_shape),
            code=qform_code,
        )
    else:
        output_image.set_qform(None, code=0)

    if sform_code > 0 and sform is not None:
        output_image.set_sform(
            _scaled_affine(sform, original_shape, output_shape),
            code=sform_code,
        )
    elif qform_code > 0:
        output_image.set_sform(None, code=0)
    else:
        # A valid transform is needed after reload; code 2 denotes an aligned
        # coordinate space when the source did not define qform or sform.
        output_image.set_sform(target_affine, code=2)

    finite_values = output_data[np.isfinite(output_data)]
    if finite_values.size:
        output_image.header["cal_min"] = float(finite_values.min())
        output_image.header["cal_max"] = float(finite_values.max())
    return output_image


def _atomic_save(
    image,
    destination: Path,
    *,
    file_mode: int,
    overwrite: bool,
) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".nii.gz",
        dir=str(destination.parent),
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        nib.save(image, str(temporary_path))
        os.chmod(temporary_path, file_mode)
        if overwrite:
            os.replace(temporary_path, destination)
        else:
            try:
                # The temporary file and destination share a directory, so this
                # hard-link commit is same-filesystem and fails atomically if a
                # concurrent process created the destination first.
                os.link(temporary_path, destination)
            except FileExistsError as exc:
                raise FileExistsError(
                    f"Output file already exists: {destination}. "
                    "Use overwrite=True to replace it."
                ) from exc
            temporary_path.unlink()
    finally:
        temporary_path.unlink(missing_ok=True)


def _validate_input_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Input file does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"Input path is not a file: {path}")
    if not _has_nii_gz_suffix(path):
        raise ValueError(f"Input file must end in .nii.gz: {path}")


def _validate_output_file(source: Path, destination: Path, *, overwrite: bool) -> None:
    if not _has_nii_gz_suffix(destination):
        raise ValueError(f"Output file must end in .nii.gz: {destination}")
    if source.resolve() == destination.resolve():
        raise ValueError("Input and output files must be different.")
    if destination.exists():
        if destination.is_dir():
            raise IsADirectoryError(f"Output path is a directory: {destination}")
        if not overwrite:
            raise FileExistsError(
                f"Output file already exists: {destination}. Use overwrite=True to replace it."
            )


def _has_nii_gz_suffix(path: Path) -> bool:
    return path.name.casefold().endswith(".nii.gz")


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True
