#!/usr/bin/env python3

"""Render a simple 3D isosurface preview for a raw density volume."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage.measure import marching_cubes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a 3D preview of a raw density volume.")
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
        "--iso",
        type=float,
        nargs="+",
        default=[0.10, 0.22],
        help="One or more normalized isosurface levels.",
    )
    return parser.parse_args()


def load_volume(path: Path, resolution: tuple[int, int, int], fmt: str) -> np.ndarray:
    nx, ny, nz = resolution
    voxel_count = nx * ny * nz
    dtype = np.uint8 if fmt == "u8" else np.float32
    data = np.fromfile(path, dtype=dtype, count=voxel_count)
    if data.size != voxel_count:
        raise ValueError(
            f"Expected {voxel_count} voxels for resolution {resolution}, found {data.size}"
        )
    if fmt == "u8":
        data = data.astype(np.float32) / 255.0
    else:
        data = data.astype(np.float32, copy=False)
    return data.reshape((nz, ny, nx))


def normalize_volume(volume: np.ndarray) -> np.ndarray:
    volume = np.clip(volume, 0.0, None)
    peak = float(volume.max())
    if peak > 1e-8:
        volume = volume / peak
    return volume


def voxel_to_world(verts: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
    nz, ny, nx = shape
    out = np.empty_like(verts, dtype=np.float32)
    out[:, 0] = (verts[:, 2] / max(nx - 1, 1)) * 2.0 - 1.0
    out[:, 1] = (verts[:, 1] / max(ny - 1, 1)) * 2.0 - 1.0
    out[:, 2] = (verts[:, 0] / max(nz - 1, 1)) * 2.0 - 1.0
    return out


def add_isosurface(ax, volume: np.ndarray, level: float, color: tuple[float, float, float], alpha: float) -> bool:
    if not np.any(volume >= level):
        return False

    verts, faces, _normals, _values = marching_cubes(volume, level=level)
    verts = voxel_to_world(verts, volume.shape)

    mesh = Poly3DCollection(verts[faces], linewidths=0.05)
    mesh.set_facecolor((*color, alpha))
    mesh.set_edgecolor((0.0, 0.0, 0.0, min(alpha * 0.25, 0.15)))
    ax.add_collection3d(mesh)
    return True


def draw_bbox(ax) -> None:
    mins = np.array([-1.0, -1.0, -1.0], dtype=np.float32)
    maxs = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    corners = np.array(
        [
            [mins[0], mins[1], mins[2]],
            [maxs[0], mins[1], mins[2]],
            [maxs[0], maxs[1], mins[2]],
            [mins[0], maxs[1], mins[2]],
            [mins[0], mins[1], maxs[2]],
            [maxs[0], mins[1], maxs[2]],
            [maxs[0], maxs[1], maxs[2]],
            [mins[0], maxs[1], maxs[2]],
        ],
        dtype=np.float32,
    )
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    for a, b in edges:
        xs = [corners[a, 0], corners[b, 0]]
        ys = [corners[a, 1], corners[b, 1]]
        zs = [corners[a, 2], corners[b, 2]]
        ax.plot(xs, ys, zs, color=(0.75, 0.75, 0.75), linewidth=0.6, alpha=0.45)


def render_preview(volume: np.ndarray, output: Path, levels: list[float]) -> None:
    volume = normalize_volume(volume)
    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111, projection="3d")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    colors = [
        (0.25, 0.25, 0.30),
        (0.55, 0.62, 0.72),
        (0.90, 0.73, 0.42),
    ]
    alphas = [0.12, 0.28, 0.42]

    any_surface = False
    for idx, level in enumerate(levels):
        ok = add_isosurface(
            ax,
            volume,
            level=level,
            color=colors[min(idx, len(colors) - 1)],
            alpha=alphas[min(idx, len(alphas) - 1)],
        )
        any_surface = any_surface or ok

    if not any_surface:
        raise RuntimeError("No isosurface could be extracted at the requested level(s).")

    draw_bbox(ax)
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-1.0, 1.0)
    ax.set_zlim(-1.0, 1.0)
    ax.set_box_aspect((1.0, 1.0, 1.0))
    ax.view_init(elev=18, azim=-54)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(
        f"3D Smoke Isosurface\n"
        f"shape={volume.shape[2]}x{volume.shape[1]}x{volume.shape[0]}  "
        f"levels={', '.join(f'{l:.2f}' for l in levels)}",
        pad=18,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output = args.output
    if output is None:
        stem = args.input.with_suffix("")
        output = stem.parent / f"{stem.name}_3d.png"

    volume = load_volume(args.input, tuple(args.resolution), args.format)
    render_preview(volume, output, args.iso)
    print(f"Wrote 3D volume preview to {output}")


if __name__ == "__main__":
    main()
