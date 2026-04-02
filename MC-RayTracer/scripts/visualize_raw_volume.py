#!/usr/bin/env python3

"""Create a quick preview image for a raw density volume."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize a raw density volume.")
    parser.add_argument("input", type=Path, help="Path to the raw density file.")
    parser.add_argument(
        "--resolution",
        type=int,
        nargs=3,
        required=True,
        metavar=("NX", "NY", "NZ"),
        help="Grid resolution.",
    )
    parser.add_argument(
        "--format",
        choices=("u8", "f32"),
        default="f32",
        help="Raw voxel format.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output preview PNG path.",
    )
    parser.add_argument(
        "--cmap",
        default="magma",
        help="Matplotlib colormap for density visualization.",
    )
    return parser.parse_args()


def load_volume(path: Path, resolution: tuple[int, int, int], fmt: str) -> np.ndarray:
    nx, ny, nz = resolution
    voxel_count = nx * ny * nz

    if fmt == "u8":
        data = np.fromfile(path, dtype=np.uint8, count=voxel_count).astype(np.float32) / 255.0
    else:
        data = np.fromfile(path, dtype=np.float32, count=voxel_count)

    if data.size != voxel_count:
        raise ValueError(
            f"Expected {voxel_count} voxels for resolution {resolution}, found {data.size}"
        )

    return data.reshape((nz, ny, nx))


def normalize_for_display(volume: np.ndarray) -> np.ndarray:
    volume = np.clip(volume, 0.0, None)
    hi = np.quantile(volume, 0.995)
    if hi > 1e-8:
        volume = np.clip(volume / hi, 0.0, 1.0)
    return np.power(volume, 0.65)


def make_preview(volume: np.ndarray, output: Path, cmap: str) -> None:
    nz, ny, nx = volume.shape
    cx, cy, cz = nx // 2, ny // 2, nz // 2

    vol_disp = normalize_for_display(volume)

    xy_slice = vol_disp[cz, :, :]
    xz_slice = vol_disp[:, cy, :]
    yz_slice = vol_disp[:, :, cx]

    xy_mip = vol_disp.max(axis=0)
    xz_mip = vol_disp.max(axis=1)
    yz_mip = vol_disp.max(axis=2)

    panels = [
        ("XY Center Slice", xy_slice),
        ("XZ Center Slice", xz_slice),
        ("YZ Center Slice", yz_slice),
        ("XY Max Projection", xy_mip),
        ("XZ Max Projection", xz_mip),
        ("YZ Max Projection", yz_mip),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True)
    for ax, (title, image) in zip(axes.flat, panels):
        ax.imshow(image, cmap=cmap, origin="lower")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(
        f"{output.stem}\nshape={nx}x{ny}x{nz}  min={volume.min():.4f}  "
        f"mean={volume.mean():.4f}  max={volume.max():.4f}",
        fontsize=12,
    )
    fig.savefig(output, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output = args.output
    if output is None:
        output = args.input.with_suffix("")
        output = output.parent / f"{output.name}_preview.png"

    volume = load_volume(args.input, tuple(args.resolution), args.format)
    output.parent.mkdir(parents=True, exist_ok=True)
    make_preview(volume, output, args.cmap)
    print(f"Wrote volume preview to {output}")


if __name__ == "__main__":
    main()
