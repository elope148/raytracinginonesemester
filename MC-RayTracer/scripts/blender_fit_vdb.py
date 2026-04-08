#!/usr/bin/env python3

"""Import a VDB in Blender, center it, and scale it to fit a target box.

Run with Blender:

    blender --background --python scripts/blender_fit_vdb.py -- \
        --input assets/volumes/foo.vdb \
        --output-json /tmp/foo_fit.json \
        --target-max-dim 2.0

The script reports the imported local AABB, computes a uniform scale so the
largest dimension matches ``target_max_dim``, and translates the volume so the
scaled bounding-box center lands on ``target_center``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import bpy
from mathutils import Vector


def parse_args() -> argparse.Namespace:
    argv = []
    if "--" in __import__("sys").argv:
        argv = __import__("sys").argv[__import__("sys").argv.index("--") + 1 :]

    parser = argparse.ArgumentParser(description="Center and scale a VDB in Blender.")
    parser.add_argument("--input", type=Path, required=True, help="Input VDB file.")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional JSON report path.",
    )
    parser.add_argument(
        "--target-max-dim",
        type=float,
        default=2.0,
        help="Uniformly scale so the largest axis equals this size.",
    )
    parser.add_argument(
        "--target-center",
        type=float,
        nargs=3,
        default=(0.0, 0.0, 0.0),
        metavar=("X", "Y", "Z"),
        help="World-space center for the fitted bounding box.",
    )
    parser.add_argument(
        "--save-blend",
        type=Path,
        default=None,
        help="Optional .blend file path to save the imported and transformed scene.",
    )
    return parser.parse_args(argv)


def vec_min(a: Vector, b: Vector) -> Vector:
    return Vector((min(a.x, b.x), min(a.y, b.y), min(a.z, b.z)))


def vec_max(a: Vector, b: Vector) -> Vector:
    return Vector((max(a.x, b.x), max(a.y, b.y), max(a.z, b.z)))


def compute_local_bbox(obj: bpy.types.Object) -> tuple[Vector, Vector]:
    corners = [Vector(corner) for corner in obj.bound_box]
    bb_min = corners[0].copy()
    bb_max = corners[0].copy()
    for corner in corners[1:]:
        bb_min = vec_min(bb_min, corner)
        bb_max = vec_max(bb_max, corner)
    return bb_min, bb_max


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"Input VDB not found: {args.input}")

    bpy.ops.wm.read_factory_settings(use_empty=True)
    res = bpy.ops.object.volume_import(filepath=str(args.input))
    if "FINISHED" not in res:
        raise RuntimeError(f"Blender failed to import {args.input}")

    obj = bpy.context.active_object
    if obj is None:
        raise RuntimeError("No active object after VDB import.")

    bb_min, bb_max = compute_local_bbox(obj)
    bbox_center = 0.5 * (bb_min + bb_max)
    bbox_dims = bb_max - bb_min
    max_dim = max(bbox_dims.x, bbox_dims.y, bbox_dims.z)
    if max_dim <= 1e-8:
        raise RuntimeError("Imported VDB has degenerate dimensions.")

    uniform_scale = args.target_max_dim / max_dim
    target_center = Vector(args.target_center)

    obj.scale = Vector((uniform_scale, uniform_scale, uniform_scale))
    obj.location = target_center - uniform_scale * bbox_center

    fitted_min = obj.location + uniform_scale * bb_min
    fitted_max = obj.location + uniform_scale * bb_max

    report = {
        "input_vdb": str(args.input),
        "object_name": obj.name,
        "original_bbox_min": list(bb_min),
        "original_bbox_max": list(bb_max),
        "original_dimensions": list(bbox_dims),
        "bbox_center": list(bbox_center),
        "uniform_scale": uniform_scale,
        "target_center": list(target_center),
        "fitted_bbox_min": list(fitted_min),
        "fitted_bbox_max": list(fitted_max),
        "fitted_dimensions": list(fitted_max - fitted_min),
    }

    print(json.dumps(report, indent=2))

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    if args.save_blend is not None:
        args.save_blend.parent.mkdir(parents=True, exist_ok=True)
        bpy.ops.wm.save_as_mainfile(filepath=str(args.save_blend))


if __name__ == "__main__":
    main()
