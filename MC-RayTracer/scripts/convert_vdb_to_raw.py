#!/usr/bin/env python3

"""Convert a VDB scalar grid into the renderer's raw volume format.

This script is intended to be run with the Python interpreter from the
`turbulence_renderer` conda env, where `pyopenvdb` is installed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyopenvdb
from scipy.ndimage import zoom


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert a VDB scalar grid to raw.")
    parser.add_argument("input", type=Path, help="Input VDB file.")
    parser.add_argument(
        "--grid",
        default="density",
        help="Grid name to extract from the VDB file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output raw file path. Defaults next to the input VDB.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        nargs=3,
        default=None,
        metavar=("NX", "NY", "NZ"),
        help="Explicit output resolution.",
    )
    parser.add_argument(
        "--max-dim",
        type=int,
        default=160,
        help="If --resolution is omitted, scale so the largest axis matches this value.",
    )
    parser.add_argument(
        "--normalize-percentile",
        type=float,
        default=99.5,
        help="Percentile used to normalize the dense grid before export.",
    )
    parser.add_argument(
        "--density-scale",
        type=float,
        default=1.0,
        help="Extra multiplier applied after normalization.",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=None,
        help="Optional JSON metadata sidecar path.",
    )
    parser.add_argument(
        "--bbox-min",
        type=int,
        nargs=3,
        default=None,
        metavar=("IX", "IY", "IZ"),
        help="Optional voxel-space bounding-box min to sample from.",
    )
    parser.add_argument(
        "--bbox-max",
        type=int,
        nargs=3,
        default=None,
        metavar=("AX", "AY", "AZ"),
        help="Optional voxel-space bounding-box max to sample from.",
    )
    parser.add_argument(
        "--swap-yz",
        action="store_true",
        help="Swap source Y/Z axes so Blender Z-up data becomes renderer Y-up data.",
    )
    return parser.parse_args()


def bbox_dims(bbox: tuple[tuple[int, int, int], tuple[int, int, int]]) -> tuple[int, int, int]:
    return tuple(int(bbox[1][axis] - bbox[0][axis] + 1) for axis in range(3))


def read_grid(
    path: Path,
    grid_name: str,
    requested_bbox: tuple[tuple[int, int, int], tuple[int, int, int]] | None = None,
) -> tuple[np.ndarray, tuple[tuple[int, int, int], tuple[int, int, int]], tuple[tuple[int, int, int], tuple[int, int, int]]]:
    grid = pyopenvdb.read(str(path), grid_name)
    active_bbox = grid.evalActiveVoxelBoundingBox()
    bbox = requested_bbox or active_bbox
    dim = bbox_dims(bbox)
    data_xyz = np.zeros(dim, dtype=np.float32)
    grid.copyToArray(data_xyz, ijk=bbox[0])
    return data_xyz, bbox, active_bbox


def choose_resolution(src_xyz: tuple[int, int, int], requested: tuple[int, int, int] | None, max_dim: int) -> tuple[int, int, int]:
    if requested is not None:
        return tuple(int(v) for v in requested)

    src = np.array(src_xyz, dtype=np.float32)
    scale = float(max_dim) / float(np.max(src))
    out = np.maximum(1, np.rint(src * scale)).astype(np.int32)
    return int(out[0]), int(out[1]), int(out[2])


def resample_xyz(volume_xyz: np.ndarray, out_xyz: tuple[int, int, int]) -> np.ndarray:
    if volume_xyz.shape == out_xyz:
        return volume_xyz.astype(np.float32, copy=False)

    factors = [float(o) / float(i) for o, i in zip(out_xyz, volume_xyz.shape)]
    return zoom(volume_xyz, zoom=factors, order=1, mode="nearest").astype(np.float32, copy=False)


def normalize_density(volume_xyz: np.ndarray, percentile: float, density_scale: float) -> np.ndarray:
    volume_xyz = np.clip(volume_xyz, 0.0, None)
    nonzero = volume_xyz[volume_xyz > 0.0]
    reference_values = nonzero if nonzero.size else volume_xyz.reshape(-1)
    ref = float(np.percentile(reference_values, percentile))
    if ref > 1e-8:
        volume_xyz = volume_xyz / ref
    volume_xyz = np.clip(volume_xyz * density_scale, 0.0, 1.0)
    return volume_xyz.astype(np.float32, copy=False)


def xyz_to_raw_layout(volume_xyz: np.ndarray) -> np.ndarray:
    # The renderer expects flattened indexing with x varying fastest:
    # idx = (z * ny + y) * nx + x. In NumPy terms that is a (nz, ny, nx)
    # array written in row-major order.
    return np.transpose(volume_xyz, (2, 1, 0)).astype(np.float32, copy=False)


def reorient_xyz(volume_xyz: np.ndarray, *, swap_yz: bool) -> np.ndarray:
    if swap_yz:
        volume_xyz = np.transpose(volume_xyz, (0, 2, 1))
    return volume_xyz.astype(np.float32, copy=False)


def default_output_path(input_path: Path, grid_name: str, out_xyz: tuple[int, int, int]) -> Path:
    stem = input_path.with_suffix("")
    nx, ny, nz = out_xyz
    return stem.parent / f"{stem.name}_{grid_name}_{nx}x{ny}x{nz}_f32.raw"


def write_metadata(
    metadata_path: Path,
    *,
    input_path: Path,
    grid_name: str,
    bbox: tuple[tuple[int, int, int], tuple[int, int, int]],
    active_bbox: tuple[tuple[int, int, int], tuple[int, int, int]],
    src_xyz: tuple[int, int, int],
    out_xyz: tuple[int, int, int],
        density_stats: dict[str, float],
        axis_transform: str,
) -> None:
    payload = {
        "input_vdb": str(input_path),
        "grid": grid_name,
        "sampled_voxel_bbox_min": list(bbox[0]),
        "sampled_voxel_bbox_max": list(bbox[1]),
        "active_voxel_bbox_min": list(active_bbox[0]),
        "active_voxel_bbox_max": list(active_bbox[1]),
        "source_resolution_xyz": list(src_xyz),
        "output_resolution_xyz": list(out_xyz),
        "axis_transform": axis_transform,
        "raw_layout": "zyx_row_major",
        "stats": density_stats,
    }
    metadata_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"Input VDB not found: {args.input}")

    requested_bbox = None
    if args.bbox_min is not None or args.bbox_max is not None:
        if args.bbox_min is None or args.bbox_max is None:
            raise ValueError("Use both --bbox-min and --bbox-max together.")
        requested_bbox = (
            tuple(int(v) for v in args.bbox_min),
            tuple(int(v) for v in args.bbox_max),
        )

    volume_xyz, bbox, active_bbox = read_grid(args.input, args.grid, requested_bbox=requested_bbox)
    src_xyz = tuple(int(v) for v in volume_xyz.shape)
    volume_xyz = reorient_xyz(volume_xyz, swap_yz=args.swap_yz)
    transformed_xyz = tuple(int(v) for v in volume_xyz.shape)
    out_xyz = choose_resolution(transformed_xyz, args.resolution, args.max_dim)
    volume_xyz = resample_xyz(volume_xyz, out_xyz)
    volume_xyz = normalize_density(volume_xyz, args.normalize_percentile, args.density_scale)

    raw_volume = xyz_to_raw_layout(volume_xyz)

    output_path = args.output or default_output_path(args.input, args.grid, out_xyz)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    raw_volume.tofile(output_path)

    stats = {
        "min": float(volume_xyz.min()),
        "mean": float(volume_xyz.mean()),
        "max": float(volume_xyz.max()),
        "nonzero_fraction": float(np.count_nonzero(volume_xyz > 0.0) / volume_xyz.size),
    }

    metadata_path = args.metadata
    if metadata_path is None:
        metadata_path = output_path.with_suffix(".json")
    write_metadata(
        metadata_path,
        input_path=args.input,
        grid_name=args.grid,
        bbox=bbox,
        active_bbox=active_bbox,
        src_xyz=src_xyz,
        out_xyz=out_xyz,
        density_stats=stats,
        axis_transform="swap_yz" if args.swap_yz else "identity",
    )

    print(f"Input VDB: {args.input}")
    print(f"Grid: {args.grid}")
    print(f"Active bbox: {bbox}")
    print(f"Source resolution (xyz): {src_xyz}")
    print(f"Transformed resolution (xyz): {transformed_xyz}")
    print(f"Output resolution (xyz): {out_xyz}")
    print(f"Raw output: {output_path}")
    print(f"Metadata: {metadata_path}")
    print(
        "Density stats: "
        f"min={stats['min']:.5f}, mean={stats['mean']:.5f}, "
        f"max={stats['max']:.5f}, nonzero_fraction={stats['nonzero_fraction']:.5f}"
    )


if __name__ == "__main__":
    main()
