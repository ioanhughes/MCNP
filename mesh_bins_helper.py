#!/usr/bin/env python3
"""
mesh_bins_helper.py
-------------------
Compute uniform-voxel mesh bin counts (n_x, n_y, n_z) for a 3D region so that
spacing is the same on all axes, given target voxel size or an exact-ratio mode.

Usage (CLI examples)
--------------------
1) Uniform spacing near a target voxel size (default mode):
   python mesh_bins_helper.py --xmin 0 --xmax 6.0 --ymin 0 --ymax 4.0 --zmin 0 --zmax 3.0 \
       --delta 0.25

2) Exact-ratio mode (forces identical spacing exactly by scaling integer ratios):
   python mesh_bins_helper.py --xmin 0 --xmax 6.0 --ymin 0 --ymax 4.0 --zmin 0 --zmax 3.0 \
       --delta 0.25 --mode ratio --max-denominator 1000

3) Also emit FMESH-style helper text (bounds + counts):
   python mesh_bins_helper.py ... --emit-fmesh --particle n --quantity flux

Notes
-----
- Units are arbitrary but consistent (e.g., cm).
- In 'uniform' mode, counts are rounded directly from target Δ, resulting in near-equal but not forced-identical spacing.
- In 'ratio' mode, axis lengths are expressed as rational ratios; spacing is identical by
  construction with Δ = L_min / n0, where n0 = ceil(L_min / delta).

Output
------
A JSON summary is printed to stdout, optionally followed by an FMESH helper block.
"""

from __future__ import annotations
import argparse
import json
from dataclasses import dataclass, asdict
from fractions import Fraction
from math import ceil, isfinite
from typing import Tuple, Dict, List, Optional


@dataclass
class Extents:
    xmin: float
    xmax: float
    ymin: float
    ymax: float
    zmin: float
    zmax: float

    @property
    def Lx(self) -> float:
        return self.xmax - self.xmin

    @property
    def Ly(self) -> float:
        return self.ymax - self.ymin

    @property
    def Lz(self) -> float:
        return self.zmax - self.zmin

    def validate(self) -> None:
        for name, lo, hi in [
            ("x", self.xmin, self.xmax),
            ("y", self.ymin, self.ymax),
            ("z", self.zmin, self.zmax),
        ]:
            if not (isfinite(lo) and isfinite(hi)):
                raise ValueError(f"{name}-bounds must be finite numbers.")
            if hi <= lo:
                raise ValueError(f"{name}max must be > {name}min (got {lo} .. {hi}).")


@dataclass
class MeshResult:
    nx: int
    ny: int
    nz: int
    delta: float
    delta_x: float
    delta_y: float
    delta_z: float
    total_voxels: int

    @property
    def shape(self) -> Tuple[int, int, int]:
        return (self.nx, self.ny, self.nz)

    # Convenience aliases using MCNP FMESH nomenclature
    @property
    def iints(self) -> int:
        return self.nx

    @property
    def jints(self) -> int:
        return self.ny

    @property
    def kints(self) -> int:
        return self.nz


def _round_positive(x: float) -> int:
    # Standard 'round' can go to even; use int(floor(x+0.5)) equivalently via Python's round
    # but ensure at least 1.
    n = int(round(x))
    return max(1, n)


def compute_uniform(ext: Extents, delta_target: float) -> MeshResult:
    """Uniform spacing near a target Δ; counts are rounded directly from target Δ, resulting in near-equal but not forced-identical spacing."""
    if delta_target <= 0:
        raise ValueError("delta_target must be > 0.")
    Lx, Ly, Lz = ext.Lx, ext.Ly, ext.Lz

    # Directly round counts from target delta
    nx = _round_positive(Lx / delta_target)
    ny = _round_positive(Ly / delta_target)
    nz = _round_positive(Lz / delta_target)

    delta_x = Lx / nx
    delta_y = Ly / ny
    delta_z = Lz / nz

    delta_exact = (delta_x + delta_y + delta_z) / 3

    return MeshResult(
        nx=nx, ny=ny, nz=nz,
        delta=delta_exact,
        delta_x=delta_x,
        delta_y=delta_y,
        delta_z=delta_z,
        total_voxels=nx * ny * nz
    )


def compute_ratio(ext: Extents, delta_min: float, max_denominator: int = 1000) -> MeshResult:
    """
    Exact-ratio mode:
      1) Express Lx:Ly:Lz as rational ratios with limited denominators.
      2) Choose n0 = ceil(Lmin / delta_min).
      3) Set nx = n0*px, ny = n0*py, nz = n0*pz, where px:py:pz are the integer ratio parts.
      4) Δ = Lmin / n0 is identical on all axes by construction.
    """
    if delta_min <= 0:
        raise ValueError("delta_min must be > 0.")
    Lx, Ly, Lz = ext.Lx, ext.Ly, ext.Lz
    Lmin = min(Lx, Ly, Lz)

    # Rational approximations of ratios to the smallest length
    rx = Fraction(Lx / Lmin).limit_denominator(max_denominator)
    ry = Fraction(Ly / Lmin).limit_denominator(max_denominator)
    rz = Fraction(Lz / Lmin).limit_denominator(max_denominator)

    # Common denominator to convert to integer proportionalities
    den = rx.denominator
    den = den * ry.denominator // gcd(den, ry.denominator)
    den = den * rz.denominator // gcd(den, rz.denominator)

    px = rx.numerator * (den // rx.denominator)
    py = ry.numerator * (den // ry.denominator)
    pz = rz.numerator * (den // rz.denominator)

    # Smallest base count to achieve at most delta_min spacing on the smallest axis
    n0 = max(1, ceil(Lmin / delta_min))

    nx, ny, nz = n0 * px, n0 * py, n0 * pz
    delta_exact = Lmin / n0
    return MeshResult(nx=nx, ny=ny, nz=nz, delta=delta_exact, delta_x=delta_exact, delta_y=delta_exact, delta_z=delta_exact, total_voxels=nx * ny * nz)


def gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return abs(a)


def edges_from_counts(lo: float, hi: float, n: int) -> List[float]:
    step = (hi - lo) / n
    return [lo + i * step for i in range(n + 1)]


def emit_fmesh_helper(ext: Extents, res: MeshResult, particle: str = "n", quantity: str = "flux") -> str:
    """
    Emit a *helper* block with bounds, counts, and edges for an MCNP FMESH tally.
    This avoids committing to a single MCNP syntax variant and gives all numbers you need.
    """
    X_edges = edges_from_counts(ext.xmin, ext.xmax, res.nx)
    Y_edges = edges_from_counts(ext.ymin, ext.ymax, res.ny)
    Z_edges = edges_from_counts(ext.zmin, ext.zmax, res.nz)

    lines = []
    lines.append("# ---- FMESH helper (cartesian) ----")
    lines.append(f"# particle: {particle}    quantity: {quantity}")
    lines.append(f"# extents: x=[{ext.xmin}, {ext.xmax}]  y=[{ext.ymin}, {ext.ymax}]  z=[{ext.zmin}, {ext.zmax}]")
    lines.append(f"# counts:  nx={res.nx}  ny={res.ny}  nz={res.nz}   Δ={res.delta:.6g}")
    lines.append("# Edges (copy into your preferred FMESH syntax):")
    lines.append("X-edges = " + " ".join(f"{v:.6g}" for v in X_edges))
    lines.append("Y-edges = " + " ".join(f"{v:.6g}" for v in Y_edges))
    lines.append("Z-edges = " + " ".join(f"{v:.6g}" for v in Z_edges))
    lines.append("# -----------------------------------")
    return "\n".join(lines)


def plan_mesh(
    xmin: float, xmax: float, ymin: float, ymax: float, zmin: float, zmax: float,
    delta: float, mode: str = "uniform", max_denominator: int = 1000,
    max_voxels: Optional[int] = None, emit_fmesh: bool = False,
    particle: str = "n", quantity: str = "flux"
) -> Dict:
    ext = Extents(xmin, xmax, ymin, ymax, zmin, zmax)
    ext.validate()

    if mode not in {"uniform", "ratio"}:
        raise ValueError("mode must be 'uniform' or 'ratio'.")

    if mode == "uniform":
        res = compute_uniform(ext, delta_target=delta)
    else:
        res = compute_ratio(ext, delta_min=delta, max_denominator=max_denominator)

    warn = None
    if max_voxels is not None and res.total_voxels > max_voxels:
        warn = f"Total voxels {res.total_voxels} exceed limit {max_voxels}. Consider coarser Δ or sub-meshing."

    out = {
        "extents": asdict(ext),
        "mode": mode,
        "result": {
            **asdict(res),
            "delta_x": res.delta_x,
            "delta_y": res.delta_y,
            "delta_z": res.delta_z,
            "iints": res.iints,
            "jints": res.jints,
            "kints": res.kints,
        },
        "edges": {
            "x": edges_from_counts(ext.xmin, ext.xmax, res.nx),
            "y": edges_from_counts(ext.ymin, ext.ymax, res.ny),
            "z": edges_from_counts(ext.zmin, ext.zmax, res.nz),
        },
    }
    if warn:
        out["warning"] = warn

    if emit_fmesh:
        out["fmesh_helper"] = emit_fmesh_helper(ext, res, particle=particle, quantity=quantity)

    return out


def plan_mesh_from_origin(
    origin: Tuple[float, float, float],
    imesh: float,
    jmesh: float,
    kmesh: float,
    delta: float,
    mode: str = "uniform",
    max_denominator: int = 1000,
    max_voxels: Optional[int] = None,
    emit_fmesh: bool = False,
    particle: str = "n",
    quantity: str = "flux",
) -> Dict:
    """Convenience wrapper accepting an MCNP-style origin and IMESH/JMESH/KMESH
    end points. The origin defines the lower x/y/z corner and the IMESH/JMESH/KMESH
    numbers give the upper bounds. The returned dictionary includes suggested
    ``iints``, ``jints`` and ``kints`` values for uniform spacing."""

    xmin, ymin, zmin = origin
    return plan_mesh(
        xmin=xmin,
        xmax=imesh,
        ymin=ymin,
        ymax=jmesh,
        zmin=zmin,
        zmax=kmesh,
        delta=delta,
        mode=mode,
        max_denominator=max_denominator,
        max_voxels=max_voxels,
        emit_fmesh=emit_fmesh,
        particle=particle,
        quantity=quantity,
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Compute equal-spacing mesh bin counts for a 3D region.")
    p.add_argument("--xmin", type=float)
    p.add_argument("--xmax", type=float)
    p.add_argument("--ymin", type=float)
    p.add_argument("--ymax", type=float)
    p.add_argument("--zmin", type=float)
    p.add_argument("--zmax", type=float)
    p.add_argument("--origin", type=float, nargs=3, metavar=("OX", "OY", "OZ"), help="Lower corner of the mesh")
    p.add_argument("--imesh", type=float, help="Upper X bound (from IMESH)")
    p.add_argument("--jmesh", type=float, help="Upper Y bound (from JMESH)")
    p.add_argument("--kmesh", type=float, help="Upper Z bound (from KMESH)")
    p.add_argument("--delta", type=float, required=True, help="Target voxel size (uniform) or minimum spacing (ratio).")
    p.add_argument("--mode", choices=["uniform", "ratio"], default="uniform",
                   help="uniform: round to nearest Δ; ratio: force exact identical Δ via integer ratios.")
    p.add_argument("--max-denominator", type=int, default=1000, help="Max denominator for rational approximation (ratio mode).")
    p.add_argument("--max-voxels", type=int, default=None, help="Optional cap to flag huge meshes.")
    p.add_argument("--emit-fmesh", action="store_true", help="Emit an FMESH helper block.")
    p.add_argument("--particle", type=str, default="n")
    p.add_argument("--quantity", type=str, default="flux")

    args = p.parse_args()

    if args.origin is not None:
        if None in (args.imesh, args.jmesh, args.kmesh):
            p.error("--origin requires --imesh, --jmesh and --kmesh")
        out = plan_mesh_from_origin(
            origin=tuple(args.origin),
            imesh=args.imesh,
            jmesh=args.jmesh,
            kmesh=args.kmesh,
            delta=args.delta,
            mode=args.mode,
            max_denominator=args.max_denominator,
            max_voxels=args.max_voxels,
            emit_fmesh=args.emit_fmesh,
            particle=args.particle,
            quantity=args.quantity,
        )
    else:
        required = [args.xmin, args.xmax, args.ymin, args.ymax, args.zmin, args.zmax]
        if any(v is None for v in required):
            p.error("Must supply either xmin/xmax/ymin/ymax/zmin/zmax or origin with imesh/jmesh/kmesh")
        out = plan_mesh(
            xmin=args.xmin, xmax=args.xmax, ymin=args.ymin, ymax=args.ymax,
            zmin=args.zmin, zmax=args.zmax, delta=args.delta, mode=args.mode,
            max_denominator=args.max_denominator, max_voxels=args.max_voxels,
            emit_fmesh=args.emit_fmesh, particle=args.particle, quantity=args.quantity,
        )
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
