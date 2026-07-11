"""Command-line entry point for :mod:`nifti_downscaler`."""

from __future__ import annotations

import argparse
from typing import Optional, Sequence

from .downscale import downscale_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nifti-downscale",
        description=(
            "Downscale one 3-D .nii.gz file or a directory tree while preserving "
            "physical orientation and field of view."
        ),
    )
    parser.add_argument(
        "input",
        nargs="?",
        help="Input .nii.gz file or directory (legacy positional form).",
    )
    parser.add_argument(
        "output",
        nargs="?",
        help="Separate output file or directory (legacy positional form).",
    )

    paths = parser.add_argument_group("named input and output")
    paths.add_argument(
        "--input-folder",
        help=(
            "Folder containing .nii.gz files directly or below patient folders. "
            "An individual .nii.gz path is also accepted."
        ),
    )
    paths.add_argument(
        "--output-folder",
        help="Output folder; the input directory structure is mirrored below it.",
    )

    sizing = parser.add_mutually_exclusive_group(required=True)
    sizing.add_argument(
        "--scale-factor",
        nargs="+",
        type=float,
        metavar="FACTOR",
        help=(
            "One factor for all axes or three factors in X Y Z order. Values must "
            "be in (0, 1], with at least one value below 1."
        ),
    )
    sizing.add_argument(
        "--target-shape",
        nargs=3,
        type=int,
        metavar=("X", "Y", "Z"),
        help="Explicit output shape in NIfTI X Y Z array order.",
    )

    parser.add_argument(
        "--interpolation",
        choices=("nearest", "linear", "cubic"),
        default="linear",
        help="Voxel interpolation method (default: linear).",
    )
    parser.add_argument(
        "--no-antialias",
        action="store_true",
        help="Disable Gaussian anti-alias filtering before linear/cubic downsampling.",
    )
    parser.add_argument(
        "--non-recursive",
        action="store_true",
        help="For directory input, process only files directly inside it.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing output files. Input files are never overwritten.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    named_paths_used = args.input_folder is not None or args.output_folder is not None
    positional_paths_used = args.input is not None or args.output is not None
    if named_paths_used and positional_paths_used:
        parser.error(
            "Use either --input-folder/--output-folder or positional input/output, not both."
        )
    if named_paths_used:
        if args.input_folder is None or args.output_folder is None:
            parser.error("--input-folder and --output-folder must be used together.")
        input_path = args.input_folder
        output_path = args.output_folder
    else:
        if args.input is None or args.output is None:
            parser.error(
                "Provide --input-folder and --output-folder, or positional input and output."
            )
        input_path = args.input
        output_path = args.output

    scale_factor = args.scale_factor
    if scale_factor is not None and len(scale_factor) not in (1, 3):
        parser.error("--scale-factor accepts exactly one or three values.")
    if scale_factor is not None and len(scale_factor) == 1:
        scale_factor = scale_factor[0]

    try:
        results = downscale_path(
            input_path,
            output_path,
            scale_factor=scale_factor,
            target_shape=args.target_shape,
            interpolation=args.interpolation,
            antialias=not args.no_antialias,
            overwrite=args.overwrite,
            recursive=not args.non_recursive,
        )
    except (OSError, ValueError) as exc:
        parser.exit(1, f"error: {exc}\n")

    for result in results:
        percentage = result.voxel_fraction * 100.0
        print(
            f"{result.input_path} -> {result.output_path}: "
            f"{result.original_shape} -> {result.output_shape} "
            f"({percentage:.2f}% of voxels)"
        )
    print(f"Downscaled {len(results)} volume(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
