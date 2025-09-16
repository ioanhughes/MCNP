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

MESH_METADATA_ATTR = "_mcnp_mesh_metadata"

MATERIAL_PROPERTIES: dict[int, tuple[str, str]] = {
    1: ("Water", "0.997 g/cm^3"),
    2: ("Helium-3", "4.925e-5 atoms/cm^3"),
    3: ("Cadmium", "8.65 g/cm^3"),
    4: ("Polythene", "0.96 g/cm^3"),
    5: ("Concrete", "2.4 g/cm^3"),
    6: ("Graphite", "1.7 g/cm^3"),
    7: ("Borated Polythene", "1.04 g/cm^3"),
    8: ("Steel", "7.872 g/cm^3"),
    9: ("Wood", "0.650 g/cm^3"),
}


def _mesh_metadata_from_filename(filename: str) -> dict[str, Any] | None:
    """Return descriptive metadata extracted from an STL filename."""

    stem = os.path.splitext(os.path.basename(filename))[0]
    name_part = stem
    material_id: int | None = None
    if "_" in stem:
        candidate, suffix = stem.rsplit("_", 1)
        if suffix.isdigit():
            material_id = int(suffix)
            if material_id == 0:
                material_id = None
            else:
                name_part = candidate
    display_name = name_part.replace("_", " ").strip()
    metadata: dict[str, Any] = {}
    if display_name:
        metadata["object_name"] = display_name
    if material_id is not None:
        metadata["material_id"] = material_id
        material_info = MATERIAL_PROPERTIES.get(material_id)
        if material_info is not None:
            material_name, density = material_info
            metadata["material_name"] = material_name
            metadata["density"] = density
    if not metadata:
        return None
    metadata["file_stem"] = stem
    return metadata


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
        mesh = vedo.Mesh(full_path).alpha(1).c("lightblue").wireframe(False)
        if subdivision > 0:
            mesh.triangulate().subdivide(subdivision, method=1)
        metadata = _mesh_metadata_from_filename(file)
        if metadata:
            try:
                setattr(mesh, MESH_METADATA_ATTR, metadata)
            except Exception:  # pragma: no cover - vedo meshes may forbid attrs
                pass
        meshes.append(mesh)
    return meshes, stl_files


def mesh_to_volume(mesh: Any) -> Any:
    """Convert a :class:`vedo.Mesh` to a binary ``Volume``.

    The mesh is voxelised and converted into a ``Volume`` where voxels
    inside the mesh have value ``1`` and all others ``0``.  The result can
    subsequently be probed against another volume to sample values inside the
    mesh rather than only on its surface.
    """
    if vedo is None:  # pragma: no cover - optional dependency
        return None
    # ``voxelize`` is preferred for speed but ``tovolume`` is a valid
    # fallback on older versions of ``vedo``.
    try:  # pragma: no cover - best effort depending on vedo version
        vol = mesh.voxelize().tovolume()
    except Exception:  # pragma: no cover - fallback path
        try:
            vol = mesh.tovolume()
        except Exception:  # pragma: no cover - give up
            return None
    try:  # make sure resulting volume is binary
        vol.binarize()
    except Exception:  # pragma: no cover - not all vedo versions
        pass
    return vol


def build_volume(
    df: pd.DataFrame,
    stl_meshes: list[Any] | None,
    *,
    cmap_name: str = "jet",
    dose_quantile: float,
    log_scale: bool,
    warning_cb: Callable[[str, str], None] | None = None,
    volume_sampling: bool = False,
) -> tuple[Any, list[Any], str, float, float]:
    """Construct the ``vedo`` volume and meshes for dose mapping.

    Parameters
    ----------
    df:
        DataFrame containing the dose values on a regular mesh.
    stl_meshes:
        List of ``vedo`` mesh objects representing geometry.
    volume_sampling:
        If ``True`` each mesh is voxelised and the dose volume is probed
        within the resulting binary mask.  The mean value within the mask is
        then assigned to the mesh so that colouring reflects interior dose
        rather than surface sampling.
    """
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

    conversion_factor: float | None = None
    if "result" in df.columns:
        result_vals = df["result"].to_numpy(dtype=float, copy=False)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratios = df["dose"].to_numpy(dtype=float, copy=False) / result_vals
        if ratios.size:
            finite_mask = np.isfinite(ratios) & np.isfinite(result_vals) & (result_vals != 0)
            if finite_mask.any():
                try:
                    median = float(np.median(ratios[finite_mask]))
                except Exception:  # pragma: no cover - defensive casting
                    conversion_factor = None
                else:
                    if np.isfinite(median) and median != 0.0:
                        conversion_factor = median
                    else:
                        conversion_factor = None

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
    vol.add_scalarbar(title=bar_title, size=(300, 900), font_size=36)
    try:
        vol._mcnp_dose_metadata = {  # type: ignore[attr-defined]
            "log_scale": log_scale,
            "conversion_factor": conversion_factor,
        }
    except Exception:  # pragma: no cover - vedo objects may forbid new attrs
        pass
    if volume_sampling:
        sampled: list[Any] = []
        for mesh in stl_meshes:
            mask = mesh_to_volume(mesh)
            probe_obj = mask if mask is not None else mesh
            probe_obj.probe(vol)
            try:
                values = np.asarray(probe_obj.pointdata[0])
            except Exception:  # pragma: no cover - vedo API variations
                values = np.asarray([])
            values = values[values > 0]
            mean_val = float(values.mean()) if values.size else 0.0
            npts = getattr(mesh, "npoints", 1)
            data = np.full(npts, mean_val)
            try:
                mesh.pointdata["scalars"] = data
            except Exception:  # pragma: no cover - fallback for dummy meshes
                mesh.pointdata = {"scalars": data}
            sampled.append(mesh)
        stl_meshes = sampled
    return vol, stl_meshes, cmap_name, min_dose, max_dose


def show_dose_map(
    vol: Any,
    meshes: list[Any],
    cmap_name: str,
    min_dose: float,
    max_dose: float,
    *,
    slice_viewer: bool = True,
    volume_sampling: bool = False,
    axes: dict[str, str] = AXES_LABELS,
) -> None:
    """Render a 3-D dose map using ``vedo``."""
    point_factory = getattr(vedo, "Point", None) if vedo is not None else None
    metadata = getattr(vol, "_mcnp_dose_metadata", {})

    def _format_probe_text(evt: Any) -> str:
        base_text = ""
        if point_factory is not None and evt is not None:
            picked = getattr(evt, "picked3d", None)
            if picked is not None:
                try:
                    x, y, z = picked
                except Exception:
                    pass
                else:
                    try:
                        probed = point_factory(picked).probe(vol)
                        value = probed.pointdata[0][0]
                    except Exception:
                        pass
                    else:
                        scalar_val: float | None
                        try:
                            scalar_val = float(value)
                        except (TypeError, ValueError):
                            scalar_val = None
                        if scalar_val is not None and np.isfinite(scalar_val):
                            log_scale = bool(metadata.get("log_scale", False))
                            if log_scale:
                                try:
                                    dose_value = float(np.power(10.0, scalar_val))
                                except OverflowError:
                                    dose_value = float("inf")
                            else:
                                dose_value = scalar_val
                            if np.isfinite(dose_value):
                                parts: list[str] = []
                                conv_factor = metadata.get("conversion_factor")
                                try:
                                    conv_factor = float(conv_factor)
                                except (TypeError, ValueError):
                                    conv_factor = None
                                if (
                                    conv_factor is not None
                                    and conv_factor != 0.0
                                    and np.isfinite(conv_factor)
                                ):
                                    try:
                                        result_value = dose_value / conv_factor
                                    except Exception:
                                        result_value = None
                                    else:
                                        if result_value is not None and np.isfinite(result_value):
                                            parts.append(f"Result: {result_value:.3g}")
                                parts.append(f"Dose: {dose_value:.3g} µSv/h")
                                if log_scale:
                                    parts.append(f"log10: {scalar_val:.3g}")
                                coords = f"({x:.3g}, {y:.3g}, {z:.3g})"
                                base_text = f"{' | '.join(parts)} @ {coords}"

        metadata_text = ""
        actor = None
        if evt is not None:
            for attr in ("actor", "object", "mesh"):
                actor = getattr(evt, attr, None)
                if actor is not None:
                    break
        if actor is not None:
            mesh_metadata = getattr(actor, MESH_METADATA_ATTR, None)
            if isinstance(mesh_metadata, dict):
                info_parts: list[str] = []
                object_name = mesh_metadata.get("object_name")
                if object_name:
                    info_parts.append(f"Object: {object_name}")
                material_name = mesh_metadata.get("material_name")
                material_id = mesh_metadata.get("material_id")
                if material_name:
                    material_text = str(material_name)
                    if isinstance(material_id, int):
                        material_text = f"{material_text} ({material_id})"
                    info_parts.append(f"Material: {material_text}")
                density = mesh_metadata.get("density")
                if density:
                    info_parts.append(f"Density: {density}")
                if info_parts:
                    metadata_text = " | ".join(info_parts)

        if metadata_text and base_text:
            return f"{base_text}\n{metadata_text}"
        if metadata_text:
            return metadata_text
        return base_text

    if slice_viewer:
        if Slicer3DPlotter is None:
            raise RuntimeError("Slice viewer not available")
        plt = Slicer3DPlotter(vol, axes=axes, cmaps=['jet'], draggable=True)
        for mesh in meshes:
            if not volume_sampling:
                mesh.probe(vol)
            mesh.cmap(cmap_name, vmin=min_dose, vmax=max_dose)
            plt += mesh
        if hasattr(plt, "add"):
            annotation = Text2D("", pos="top-left", bg="w", alpha=0.5)
            plt.add(annotation)
        if hasattr(plt, "add_callback"):
            def _probe(evt: Any) -> None:
                annotation.text(_format_probe_text(evt))
                if hasattr(plt, "render"):
                    plt.render()

            plt.add_callback("MouseMove", _probe)
        if hasattr(plt, "show"):
            plt.show()
        elif hasattr(plt, "interactive"):
            plt.interactive()
    else:
        for mesh in meshes:
            if not volume_sampling:
                mesh.probe(vol)
            mesh.cmap(cmap_name, vmin=min_dose, vmax=max_dose)
        plt = show(vol, meshes, axes=axes, interactive=False)
        if hasattr(plt, "add_callback"):
            annotation = Text2D("", pos="top-left", bg="w", alpha=0.5)
            plt.add(annotation)

            def _probe(evt: Any) -> None:
                annotation.text(_format_probe_text(evt))
                if hasattr(plt, "render"):
                    plt.render()

            plt.add_callback("MouseMove", _probe)
        if hasattr(plt, "interactive"):
            plt.interactive()
