"""Utilities for 3-D plotting with ``vedo``."""
from __future__ import annotations

import os
from typing import Any, Callable

import numpy as np
import pandas as pd

try:  # pragma: no cover - optional dependency
    import vedo  # type: ignore
    from vedo import Volume, show, Text2D  # type: ignore
    from vedo.applications import Slicer3DPlotter  # type: ignore
except Exception:  # pragma: no cover - vedo not available
    vedo = None  # type: ignore[assignment]
    Volume = None  # type: ignore[assignment]
    show = None  # type: ignore[assignment]
    Slicer3DPlotter = None  # type: ignore[assignment]

AXES_LABELS = {"xTitle": "x (cm)", "yTitle": "y (cm)", "zTitle": "z (cm)"}


def load_stl_meshes(folderpath: str, subdivision: int = 0) -> tuple[list[Any], list[str]]:
    """Load all STL files from *folderpath* as ``vedo`` meshes.

    Parameters
    ----------
    folderpath:
        Directory containing STL files to load.
    subdivision:
        Optional subdivision level applied to each loaded mesh. A value of
        ``0`` leaves the mesh unchanged.
    """
    if vedo is None:  # pragma: no cover - optional dependency
        return [], []
    files_in_folder = os.listdir(folderpath)
    stl_files = [f for f in files_in_folder if f.lower().endswith(".stl")]
    meshes: list[Any] = []
    for file in stl_files:
        full_path = os.path.join(folderpath, file)
        mesh = vedo.Mesh(full_path).alpha(0.5).c("lightblue").wireframe(False)
        if subdivision > 0:
            mesh.triangulate().subdivide(subdivision)
        meshes.append(mesh)
    return meshes, stl_files


def build_volume(
    df: pd.DataFrame,
    stl_meshes: list[Any] | None,
    *,
    cmap_name: str = "jet",
    dose_quantile: float,
    log_scale: bool,
    warning_cb: Callable[[str, str], None] | None = None,
) -> tuple[Any, list[Any], str, float, float]:
    """Construct the ``vedo`` volume and meshes for dose mapping."""
    if stl_meshes is None:
        raise ValueError("No STL files loaded")

    quant = dose_quantile / 100
    max_dose = df["dose"].quantile(quant)
    if max_dose == 0:
        max_dose = 1
    min_dose = df[df["dose"] > 0]["dose"].min()
    if not pd.notna(min_dose) or min_dose <= 0:
        min_dose = max_dose / 1e6

    xs = np.sort(df["x"].unique())
    ys = np.sort(df["y"].unique())
    zs = np.sort(df["z"].unique())
    nx, ny, nz = len(xs), len(ys), len(zs)

    def _check_uniform(arr: np.ndarray, label: str) -> None:
        if len(arr) > 1:
            diffs = np.diff(arr)
            if not np.allclose(diffs, diffs[0]) and warning_cb is not None:
                warning_cb(
                    "Non-uniform mesh spacing",
                    f"{label}-coordinates are not uniformly spaced; using first spacing value.",
                )

    _check_uniform(xs, "X")
    _check_uniform(ys, "Y")
    _check_uniform(zs, "Z")

    grid = (
        df.pivot_table(index="z", columns=["y", "x"], values="dose")
        .fillna(0.0)
        .to_numpy()
        .reshape(nz, ny, nx)
        .transpose(2, 1, 0)
    )
    grid = np.clip(grid, min_dose, max_dose)

    if log_scale:
        grid = np.log10(grid)
        min_dose = np.log10(min_dose)
        max_dose = np.log10(max_dose)
        bar_title = "Log10 Dose (µSv/h)"
    else:
        bar_title = "Dose (µSv/h)"

    dx = xs[1] - xs[0] if nx > 1 else 1.0
    dy = ys[1] - ys[0] if ny > 1 else 1.0
    dz = zs[1] - zs[0] if nz > 1 else 1.0

    vol = Volume(grid, spacing=(dx, dy, dz), origin=(xs[0], ys[0], zs[0]))
    vol.cmap(cmap_name, vmin=min_dose, vmax=max_dose)
    vol.add_scalarbar(title=bar_title, size=(200, 600), font_size=24)
    return vol, stl_meshes, cmap_name, min_dose, max_dose


def show_dose_map(
    vol: Any,
    meshes: list[Any],
    cmap_name: str,
    min_dose: float,
    max_dose: float,
    *,
    slice_viewer: bool,
    axes: dict[str, str] = AXES_LABELS,
) -> None:
    """Render a 3-D dose map using ``vedo``."""
    if slice_viewer:
        if Slicer3DPlotter is None:
            raise RuntimeError("Slice viewer not available")
        plt = Slicer3DPlotter(vol, axes=axes)
        for mesh in meshes:
            mesh.probe(vol)
            mesh.cmap(cmap_name, vmin=min_dose, vmax=max_dose)
            plt += mesh
        plt.show()
    else:
        for mesh in meshes:
            mesh.probe(vol)
            mesh.cmap(cmap_name, vmin=min_dose, vmax=max_dose)
        plt = show(vol, meshes, axes=axes, interactive=False)
        if hasattr(plt, "add_callback"):
            annotation = Text2D("", pos="top-left", bg="w", alpha=0.5)
            plt.add(annotation)

            def _probe(evt: Any) -> None:
                if evt.picked3d is not None:
                    value = vedo.Point(evt.picked3d).probe(vol).pointdata[0][0]
                    annotation.text(f"{value:.3g}")
                else:
                    annotation.text("")
                plt.render()

            plt.add_callback("MouseMove", _probe)
        if hasattr(plt, "interactive"):
            plt.interactive()
