#!/usr/bin/env python3

"""Generate a Cornell-box smoke density volume as a raw grid.

This is a lightweight procedural smoke generator built on Taichi. It does not
run a full fluid solve; instead it advects a density field through a
hand-authored swirling velocity field and injects a few localized sources to
produce a plausible static smoke volume for rendering.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import taichi as ti


TI_WORLD_MIN = (-0.99, -0.99, -0.99)
TI_WORLD_MAX = (0.99, 0.99, 0.99)

MIRROR_CENTER = (-0.12, -0.58, 0.10)
MIRROR_RADIUS = 0.42

GLASS_CENTER = (0.52, -0.73, 0.28)
GLASS_RADIUS = 0.27

EMISSIVE_CENTER = (0.18, -0.82, -0.20)
EMISSIVE_RADIUS = 0.18


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a Cornell-box smoke volume.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("assets/volumes/cornell_smoke_taichi_128_f32.raw"),
        help="Output raw density file.",
    )
    parser.add_argument("--nx", type=int, default=128, help="Grid resolution along X.")
    parser.add_argument("--ny", type=int, default=128, help="Grid resolution along Y.")
    parser.add_argument("--nz", type=int, default=128, help="Grid resolution along Z.")
    parser.add_argument("--steps", type=int, default=72, help="Number of advection steps.")
    parser.add_argument("--dt", type=float, default=0.045, help="Advection timestep.")
    parser.add_argument(
        "--dissipation",
        type=float,
        default=0.992,
        help="Density dissipation per step.",
    )
    return parser.parse_args()


@ti.data_oriented
class SmokeGenerator:
    def __init__(self, nx: int, ny: int, nz: int, dt: float, dissipation: float):
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.dt = dt
        self.dissipation = dissipation

        self.density = ti.field(dtype=ti.f32, shape=(nx, ny, nz))
        self.next_density = ti.field(dtype=ti.f32, shape=(nx, ny, nz))
        self.time = ti.field(dtype=ti.f32, shape=())

    @ti.func
    def clamp01(self, x):
        return ti.min(1.0, ti.max(0.0, x))

    @ti.func
    def smoothstep(self, edge0, edge1, x):
        t = self.clamp01((x - edge0) / (edge1 - edge0 + 1e-6))
        return t * t * (3.0 - 2.0 * t)

    @ti.func
    def world_pos(self, i, j, k):
        wx = TI_WORLD_MIN[0] + (ti.cast(i, ti.f32) + 0.5) * (TI_WORLD_MAX[0] - TI_WORLD_MIN[0]) / float(self.nx)
        wy = TI_WORLD_MIN[1] + (ti.cast(j, ti.f32) + 0.5) * (TI_WORLD_MAX[1] - TI_WORLD_MIN[1]) / float(self.ny)
        wz = TI_WORLD_MIN[2] + (ti.cast(k, ti.f32) + 0.5) * (TI_WORLD_MAX[2] - TI_WORLD_MIN[2]) / float(self.nz)
        return ti.Vector([wx, wy, wz])

    @ti.func
    def sample_density(self, p):
        result = 0.0
        inside = not (
            p.x < TI_WORLD_MIN[0] or p.x > TI_WORLD_MAX[0] or
            p.y < TI_WORLD_MIN[1] or p.y > TI_WORLD_MAX[1] or
            p.z < TI_WORLD_MIN[2] or p.z > TI_WORLD_MAX[2]
        )

        if inside:
            gx = (p.x - TI_WORLD_MIN[0]) / (TI_WORLD_MAX[0] - TI_WORLD_MIN[0]) * float(self.nx - 1)
            gy = (p.y - TI_WORLD_MIN[1]) / (TI_WORLD_MAX[1] - TI_WORLD_MIN[1]) * float(self.ny - 1)
            gz = (p.z - TI_WORLD_MIN[2]) / (TI_WORLD_MAX[2] - TI_WORLD_MIN[2]) * float(self.nz - 1)

            x0 = int(ti.floor(gx))
            y0 = int(ti.floor(gy))
            z0 = int(ti.floor(gz))
            x1 = ti.min(x0 + 1, self.nx - 1)
            y1 = ti.min(y0 + 1, self.ny - 1)
            z1 = ti.min(z0 + 1, self.nz - 1)

            tx = gx - float(x0)
            ty = gy - float(y0)
            tz = gz - float(z0)

            c000 = self.density[x0, y0, z0]
            c100 = self.density[x1, y0, z0]
            c010 = self.density[x0, y1, z0]
            c110 = self.density[x1, y1, z0]
            c001 = self.density[x0, y0, z1]
            c101 = self.density[x1, y0, z1]
            c011 = self.density[x0, y1, z1]
            c111 = self.density[x1, y1, z1]

            c00 = c000 * (1.0 - tx) + c100 * tx
            c10 = c010 * (1.0 - tx) + c110 * tx
            c01 = c001 * (1.0 - tx) + c101 * tx
            c11 = c011 * (1.0 - tx) + c111 * tx
            c0 = c00 * (1.0 - ty) + c10 * ty
            c1 = c01 * (1.0 - ty) + c11 * ty
            result = c0 * (1.0 - tz) + c1 * tz

        return result

    @ti.func
    def gaussian(self, p, center, radii, sharpness):
        q = ti.Vector([
            (p.x - center.x) / radii.x,
            (p.y - center.y) / radii.y,
            (p.z - center.z) / radii.z,
        ])
        return ti.exp(-sharpness * q.dot(q))

    @ti.func
    def boundary_mask(self, p):
        dx = ti.min(p.x - TI_WORLD_MIN[0], TI_WORLD_MAX[0] - p.x)
        dy = ti.min(p.y - TI_WORLD_MIN[1], TI_WORLD_MAX[1] - p.y)
        dz = ti.min(p.z - TI_WORLD_MIN[2], TI_WORLD_MAX[2] - p.z)
        return (
            self.smoothstep(0.02, 0.14, dx) *
            self.smoothstep(0.02, 0.14, dy) *
            self.smoothstep(0.02, 0.14, dz)
        )

    @ti.func
    def sphere_clear_mask(self, p):
        d0 = (p - ti.Vector(MIRROR_CENTER)).norm()
        d1 = (p - ti.Vector(GLASS_CENTER)).norm()
        d2 = (p - ti.Vector(EMISSIVE_CENTER)).norm()

        m0 = self.smoothstep(MIRROR_RADIUS * 0.97, MIRROR_RADIUS * 1.12, d0)
        m1 = self.smoothstep(GLASS_RADIUS * 0.97, GLASS_RADIUS * 1.15, d1)
        m2 = self.smoothstep(EMISSIVE_RADIUS * 0.95, EMISSIVE_RADIUS * 1.16, d2)
        return m0 * m1 * m2

    @ti.func
    def source_density(self, p, time):
        left_sheet_center = ti.Vector([
            -0.94,
            0.48 + 0.05 * ti.sin(0.55 * time),
            -0.10 + 0.08 * ti.sin(0.31 * time),
        ])
        left_sheet = 3.1 * self.gaussian(
            p, left_sheet_center, ti.Vector([0.05, 0.22, 0.18]), 4.2
        )

        ceiling_roll_center = ti.Vector([
            -0.72 + 0.05 * ti.cos(0.22 * time),
            0.78,
            -0.36 + 0.04 * ti.sin(0.44 * time),
        ])
        ceiling_roll = 1.5 * self.gaussian(
            p, ceiling_roll_center, ti.Vector([0.18, 0.09, 0.16]), 4.8
        )

        lower_jet_center = ti.Vector([
            0.82,
            -0.72,
            0.24 + 0.05 * ti.sin(0.63 * time),
        ])
        lower_jet = 2.6 * self.gaussian(
            p, lower_jet_center, ti.Vector([0.10, 0.08, 0.12]), 6.0
        )

        return left_sheet + ceiling_roll + lower_jet

    @ti.func
    def detached_puffs(self, p):
        puff = 0.0
        puff += 0.72 * self.gaussian(
            p, ti.Vector([-0.20, 0.05, -0.10]), ti.Vector([0.12, 0.09, 0.12]), 5.2
        )
        puff += 0.80 * self.gaussian(
            p, ti.Vector([0.08, 0.04, -0.02]), ti.Vector([0.11, 0.08, 0.11]), 5.0
        )
        puff += 0.74 * self.gaussian(
            p, ti.Vector([0.28, 0.01, 0.09]), ti.Vector([0.12, 0.08, 0.10]), 5.6
        )
        puff += 0.68 * self.gaussian(
            p, ti.Vector([0.44, -0.02, 0.18]), ti.Vector([0.11, 0.07, 0.10]), 6.0
        )
        return puff

    @ti.func
    def velocity(self, p, time):
        drift = ti.Vector([0.05, 0.08, -0.01])

        left_weight = self.gaussian(
            p,
            ti.Vector([-0.92, 0.45, -0.12]),
            ti.Vector([0.35, 0.36, 0.42]),
            2.8,
        )
        left_push = left_weight * ti.Vector([0.84, 0.12, 0.08])

        ceiling_weight = self.gaussian(
            p,
            ti.Vector([-0.50, 0.78, -0.22]),
            ti.Vector([0.48, 0.22, 0.35]),
            2.3,
        )
        ceiling_shear = ceiling_weight * ti.Vector([0.34, -0.10, 0.04])

        jet_weight = self.gaussian(
            p,
            ti.Vector([0.72, -0.70, 0.24]),
            ti.Vector([0.38, 0.18, 0.22]),
            2.8,
        )
        lower_jet = jet_weight * ti.Vector([-0.90, 0.06, -0.08])

        q = p - ti.Vector(MIRROR_CENTER)
        swirl_envelope = ti.exp(-4.0 * (q.x * q.x + q.z * q.z) - 2.5 * q.y * q.y)
        mirror_swirl = swirl_envelope * ti.Vector([
            -1.35 * q.z,
            0.22 * ti.sin(2.2 * time),
            1.35 * q.x,
        ])

        noise = ti.Vector([
            0.16 * ti.sin(6.0 * p.y + 4.2 * p.z + 1.1 * time) +
            0.08 * ti.cos(5.4 * p.z - 3.8 * p.x - 0.6 * time),
            0.14 * ti.cos(5.8 * p.x - 4.6 * p.z - 0.8 * time) +
            0.08 * ti.sin(4.0 * p.x + 2.7 * p.y + 0.9 * time),
            0.18 * ti.sin(4.8 * p.x - 5.0 * p.y + 1.4 * time) -
            0.07 * ti.cos(6.2 * p.y + 3.1 * p.z - 0.7 * time),
        ])

        buoyancy = ti.Vector([0.0, 0.16 * (0.35 - p.y), 0.0])
        return drift + left_push + ceiling_shear + lower_jet + mirror_swirl + noise + buoyancy

    @ti.kernel
    def advect_step(self):
        for i, j, k in self.density:
            p = self.world_pos(i, j, k)
            time = self.time[None]
            vel0 = self.velocity(p, time)
            mid = p - 0.5 * self.dt * vel0
            vel1 = self.velocity(mid, time - 0.5 * self.dt)
            prev_p = p - self.dt * vel1

            advected = self.sample_density(prev_p) * self.dissipation
            injected = self.dt * self.source_density(p, time)

            d = (advected + injected) * self.boundary_mask(p) * self.sphere_clear_mask(p)
            self.next_density[i, j, k] = d

    @ti.kernel
    def copy_back(self):
        for I in ti.grouped(self.density):
            self.density[I] = self.next_density[I]

    @ti.kernel
    def final_shape_pass(self):
        for i, j, k in self.density:
            p = self.world_pos(i, j, k)
            d = self.density[i, j, k]
            d += 0.22 * self.detached_puffs(p)
            d *= self.boundary_mask(p) * self.sphere_clear_mask(p)
            d = ti.max(d - 0.015, 0.0)
            self.next_density[i, j, k] = ti.pow(d, 0.88)

    @ti.kernel
    def blur_pass(self):
        for i, j, k in self.density:
            accum = 0.0
            weight = 0.0
            for dx, dy, dz in ti.static(
                [
                    (0, 0, 0),
                    (-1, 0, 0), (1, 0, 0),
                    (0, -1, 0), (0, 1, 0),
                    (0, 0, -1), (0, 0, 1),
                ]
            ):
                xi = ti.min(ti.max(i + dx, 0), self.nx - 1)
                yi = ti.min(ti.max(j + dy, 0), self.ny - 1)
                zi = ti.min(ti.max(k + dz, 0), self.nz - 1)
                w = 2.0 if (dx == 0 and dy == 0 and dz == 0) else 1.0
                accum += w * self.density[xi, yi, zi]
                weight += w
            self.next_density[i, j, k] = accum / weight

    def run(self, steps: int) -> np.ndarray:
        self.density.fill(0.0)
        for step in range(steps):
            self.time[None] = step * self.dt
            self.advect_step()
            self.copy_back()

        self.final_shape_pass()
        self.copy_back()

        self.blur_pass()
        self.copy_back()

        volume = self.density.to_numpy()
        volume = np.clip(volume, 0.0, None)
        peak = float(volume.max())
        if peak > 1e-8:
            volume /= peak

        envelope = np.quantile(volume, 0.995)
        if envelope > 1e-8:
            volume = np.clip(volume / envelope, 0.0, 1.0)

        volume = np.power(volume, 1.15).astype(np.float32, copy=False)
        return volume


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    ti.init(arch=ti.cpu, default_fp=ti.f32, offline_cache=False)

    generator = SmokeGenerator(
        nx=args.nx,
        ny=args.ny,
        nz=args.nz,
        dt=args.dt,
        dissipation=args.dissipation,
    )
    volume = generator.run(args.steps)
    volume.astype(np.float32).tofile(args.output)

    nz = int(np.count_nonzero(volume > 0.02))
    print(f"Wrote smoke volume to {args.output}")
    print(f"Resolution: {args.nx} x {args.ny} x {args.nz}")
    print(
        "Density stats: "
        f"min={volume.min():.5f}, mean={volume.mean():.5f}, "
        f"max={volume.max():.5f}, occupied_voxels={nz}"
    )


if __name__ == "__main__":
    main()
