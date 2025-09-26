"""Utilities for 3-D plotting with ``vedo``."""
from __future__ import annotations

import csv
import math
import os
import re
import vtk
from importlib import resources
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

AVOGADRO_CONSTANT = 6.022_140_76e23  # atoms/mol
TRANSPARENT_ALPHA = 0.6


def _load_material_definitions() -> tuple[dict[int, dict[str, Any]], dict[str, float]]:
    """Return material metadata and molar-mass lookups from :mod:`resources`."""

    material_properties: dict[int, dict[str, Any]] = {}
    molar_mass_lookup: dict[str, float] = {}

    try:
        csv_path = resources.files(__package__).joinpath("materials.csv")
    except (AttributeError, FileNotFoundError):  # pragma: no cover - packaging issue
        return material_properties, molar_mass_lookup

    rows: list[dict[str, Any]] = []
    try:
        with csv_path.open("r", encoding="utf-8") as handle:
            df = pd.read_csv(handle, dtype=str, keep_default_na=False)
        rows = df.to_dict(orient="records")
    except Exception:  # pragma: no cover - fallback for minimal environments
        try:
            with csv_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
        except FileNotFoundError:  # pragma: no cover - defensive
            return material_properties, molar_mass_lookup

    for row in rows:
        if not isinstance(row, dict):
            continue

        material_id_raw = row.get("material_id", "")
        if material_id_raw is None:
            continue
        material_id_str = str(material_id_raw).strip()
        if not material_id_str:
            continue
        try:
            material_id = int(material_id_str)
        except ValueError:
            continue

        name = str(row.get("material_name", "") or "").strip()
        density_value = str(row.get("density_value", "") or "").strip()
        density_unit = str(row.get("density_unit", "") or "").strip()
        density_parts = [part for part in (density_value, density_unit) if part]
        if not name or not density_parts:
            continue
        density = " ".join(density_parts)
        transparent_raw = row.get("transparent", "")
        transparent = str(transparent_raw or "").strip().lower() in {"y", "yes", "true", "1"}

        material_properties[material_id] = {
            "name": name,
            "density": density,
            "transparent": transparent,
        }

        molar_mass_raw = row.get("molar_mass_g_per_mol", "")
        molar_mass_str = str(molar_mass_raw or "").strip()
        if molar_mass_str:
            try:
                molar_mass_lookup[name.lower()] = float(molar_mass_str)
            except ValueError:
                continue

    return material_properties, molar_mass_lookup


MATERIAL_PROPERTIES, MOLAR_MASS_G_PER_MOL = _load_material_definitions()


def _parse_density_value(density: str | None) -> tuple[float | None, str | None]:
    """Extract a numeric value and unit string from a density description."""

    if not isinstance(density, str):
        return None, None
    match = re.match(r"\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*(.*)", density)
    if not match:
        return None, None
    try:
        value = float(match.group(1))
    except ValueError:
        return None, None
    unit = match.group(2).strip() or None
    return value, unit


def _lookup_molar_mass(metadata: dict[str, Any] | None) -> float | None:
    """Return the molar mass for a material described by *metadata*."""

    if not isinstance(metadata, dict):
        return None

    material_name = metadata.get("material_name")
    if isinstance(material_name, str):
        molar_mass = MOLAR_MASS_G_PER_MOL.get(material_name.strip().lower())
        if molar_mass is not None:
            return molar_mass

    material_id = metadata.get("material_id")
    if isinstance(material_id, int):
        material_info = MATERIAL_PROPERTIES.get(material_id)
        if isinstance(material_info, dict):
            name = material_info.get("name")
            if isinstance(name, str):
                molar_mass = MOLAR_MASS_G_PER_MOL.get(name.lower())
                if molar_mass is not None:
                    return molar_mass

    return None


def _density_to_g_per_cm3(
    value: float | None, unit: str | None, metadata: dict[str, Any] | None = None
) -> float | None:
    """Convert a density value to ``g/cm^3`` when units are recognised."""

    if value is None or unit is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return None
    if not math.isfinite(numeric):
        return None
    unit_l = unit.lower()
    if "g/cm" in unit_l:
        return numeric
    if "kg/m" in unit_l:
        return numeric / 1000.0
    if "atom" in unit_l and "cm" in unit_l:
        molar_mass = _lookup_molar_mass(metadata)
        if molar_mass is None:
            return None
        return numeric * molar_mass / AVOGADRO_CONSTANT
    return None


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
        if isinstance(material_info, dict):
            material_name = str(material_info.get("name") or "").strip()
            density = str(material_info.get("density") or "").strip()
            if material_name:
                metadata["material_name"] = material_name
            if density:
                metadata["density"] = density
                value, unit = _parse_density_value(density)
                if value is not None:
                    metadata["density_value"] = value
                if unit:
                    metadata["density_unit"] = unit
            metadata["transparent"] = bool(material_info.get("transparent"))
    if not metadata:
        return None
    metadata["file_stem"] = stem
    return metadata


def _extract_density_in_g_cm3(metadata: dict[str, Any]) -> float | None:
    """Return density in ``g/cm^3`` for *metadata* if available."""

    density_value = metadata.get("density_value")
    density_unit = metadata.get("density_unit")

    try:
        density_value = float(density_value) if density_value is not None else None
    except (TypeError, ValueError):  # pragma: no cover - defensive
        density_value = None
    density_unit = str(density_unit) if density_unit is not None else None

    if density_value is None:
        parsed_value, parsed_unit = _parse_density_value(metadata.get("density"))
        if parsed_value is not None:
            density_value = parsed_value
            metadata.setdefault("density_value", parsed_value)
        if parsed_unit and not density_unit:
            density_unit = parsed_unit
            metadata.setdefault("density_unit", parsed_unit)

    return _density_to_g_per_cm3(density_value, density_unit, metadata)


def _fallback_voxel_indices(
    mesh: Any,
    voxel_centres: np.ndarray,
    voxel_size: tuple[float, float, float] | None,
) -> np.ndarray:
    """Return indices of voxels likely intersecting *mesh* as a fallback."""

    if vedo is None or voxel_centres.size == 0:
        return np.array([], dtype=int)

    try:
        bounds = mesh.bounds()
    except Exception:  # pragma: no cover - vedo mesh missing bounds
        return np.array([], dtype=int)

    bounds_arr = np.asarray(bounds, dtype=float)
    if bounds_arr.size != 6 or not np.all(np.isfinite(bounds_arr)):
        return np.array([], dtype=int)

    min_bounds = bounds_arr[[0, 2, 4]]
    max_bounds = bounds_arr[[1, 3, 5]]
    if np.any(max_bounds <= min_bounds):
        return np.array([], dtype=int)

    if voxel_size is None:
        voxel_size = (math.nan, math.nan, math.nan)

    dx, dy, dz = voxel_size
    spacing = [abs(dx), abs(dy), abs(dz)]
    if not all(math.isfinite(val) and val > 0.0 for val in spacing):
        spacing = [1.0, 1.0, 1.0]

    diag = math.sqrt(sum(val * val for val in spacing))
    if not math.isfinite(diag) or diag <= 0.0:
        diag = 1.0

    pad = diag * 0.5
    expanded_min = min_bounds - pad
    expanded_max = max_bounds + pad

    candidate_mask = np.all(
        (voxel_centres >= expanded_min) & (voxel_centres <= expanded_max), axis=1
    )
    if not np.any(candidate_mask):
        return np.array([], dtype=int)

    candidate_indices = np.nonzero(candidate_mask)[0]
    candidate_points = voxel_centres[candidate_indices]
    if candidate_points.size == 0:
        return np.array([], dtype=int)

    try:
        mesh.compute_normals()
    except Exception:  # pragma: no cover - optional
        pass

    distances: np.ndarray | None = None
    distance_fn = getattr(mesh, "distance_to_points", None)
    if callable(distance_fn):
        try:
            distances = np.asarray(distance_fn(candidate_points, signed=True), dtype=float)
        except Exception:  # pragma: no cover - custom distance failed
            distances = None

    if distances is None:
        try:
            point_cloud = vedo.Points(candidate_points.tolist())
            distances = np.asarray(point_cloud.distance_to(mesh, signed=True), dtype=float)
        except Exception:  # pragma: no cover - vedo failure
            return np.array([], dtype=int)

    if distances.size != candidate_indices.size:
        return np.array([], dtype=int)

    inside_mask = distances <= 0.0
    if not np.any(inside_mask):
        inside_mask = distances <= pad
    if not np.any(inside_mask):
        return np.array([], dtype=int)

    return candidate_indices[inside_mask]


def _compute_mesh_statistics(
    mesh: Any,
    voxel_centres: np.ndarray,
    dose_values: np.ndarray,
    voxel_volume: float,
    metadata: dict[str, Any],
    voxel_size: tuple[float, float, float] | None = None,
) -> dict[str, float | int]:
    """Calculate mean dose and mass-related statistics for *mesh*."""

    indices: np.ndarray | None = None
    if hasattr(mesh, "inside_points"):
        try:
            inside_ids = mesh.inside_points(voxel_centres, return_ids=True)
        except Exception:  # pragma: no cover - best-effort geometry test
            indices = None
        else:
            indices = np.asarray(inside_ids, dtype=int)
    if indices is None or indices.size == 0:
        indices = _fallback_voxel_indices(mesh, voxel_centres, voxel_size)
    if indices.size == 0:
        return {}
    indices = np.unique(indices)
    valid_mask = (indices >= 0) & (indices < dose_values.size)
    if not np.any(valid_mask):
        return {}
    indices = indices[valid_mask]
    voxel_doses = dose_values[indices]
    if voxel_doses.size == 0:
        return {}
    finite_mask = np.isfinite(voxel_doses)
    if not np.any(finite_mask):
        return {}
    voxel_doses = voxel_doses[finite_mask]
    if voxel_doses.size == 0:
        return {}

    stats: dict[str, float | int] = {
        "mean_dose_rate": float(np.mean(voxel_doses)),
        "voxel_count": int(voxel_doses.size),
    }
    if voxel_volume > 0.0 and math.isfinite(voxel_volume):
        stats["voxel_volume_cm3"] = float(voxel_volume)

    density_g = _extract_density_in_g_cm3(metadata)
    if density_g is not None and density_g > 0.0 and voxel_volume > 0.0:
        mass_per_voxel = density_g * voxel_volume
        total_mass = mass_per_voxel * voxel_doses.size
        stats["mass_per_voxel_g"] = float(mass_per_voxel)
        stats["total_mass_g"] = float(total_mass)

    return stats


def _attach_mesh_statistics(
    meshes: list[Any],
    voxel_centres: np.ndarray,
    dose_values: np.ndarray,
    voxel_volume: float,
    voxel_size: tuple[float, float, float] | None,
) -> None:
    """Annotate each mesh in *meshes* with dose statistics."""

    if voxel_centres.size == 0 or dose_values.size == 0:
        return

    for mesh in meshes:
        original_metadata = getattr(mesh, MESH_METADATA_ATTR, None)
        metadata = dict(original_metadata) if isinstance(original_metadata, dict) else {}
        stats = _compute_mesh_statistics(
            mesh, voxel_centres, dose_values, voxel_volume, metadata, voxel_size
        )
        if stats:
            existing = metadata.get("dose_statistics")
            if isinstance(existing, dict):
                merged = dict(existing)
                merged.update(stats)
            else:
                merged = stats
            metadata["dose_statistics"] = merged
        if stats or isinstance(original_metadata, dict):
            try:
                setattr(mesh, MESH_METADATA_ATTR, metadata)
            except Exception:  # pragma: no cover - vedo objects may forbid attrs
                pass

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
        mesh = vedo.Mesh(full_path).c("lightblue").wireframe(False)
        if subdivision > 0:
            mesh.triangulate().subdivide(subdivision, method=1)
        metadata = _mesh_metadata_from_filename(file)
        alpha_value = 1.0
        if metadata and metadata.get("transparent"):
            alpha_value = TRANSPARENT_ALPHA
        mesh.alpha(alpha_value)
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
    min_dose: float,
    max_dose: float,
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
    dose_quantile:
        Percentile, expressed as a percentage, used when deriving the
        normalisation bounds.
    min_dose, max_dose:
        Lower and upper bounds used for colour normalisation.
    volume_sampling:
        If ``True`` each mesh is voxelised and the dose volume is probed
        within the resulting binary mask.  The mean value within the mask is
        then assigned to the mesh so that colouring reflects interior dose
        rather than surface sampling.
    """
    if stl_meshes is None:
        raise ValueError("No STL files loaded")

    max_dose = float(max_dose)
    min_dose = float(min_dose)
    if max_dose <= 0.0:
        max_dose = 1.0
    if min_dose <= 0.0 or min_dose >= max_dose:
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

    pivot = df.pivot_table(index="z", columns=["y", "x"], values="dose").fillna(0.0)
    linear_grid = pivot.to_numpy().reshape(nz, ny, nx).transpose(2, 1, 0)
    clipped_grid = np.clip(linear_grid, min_dose, max_dose)

    dx = xs[1] - xs[0] if nx > 1 else 1.0
    dy = ys[1] - ys[0] if ny > 1 else 1.0
    dz = zs[1] - zs[0] if nz > 1 else 1.0
    voxel_volume = dx * dy * dz

    voxel_centres: np.ndarray | None = None
    dose_values: np.ndarray | None = None
    if stl_meshes:
        X, Y, Z = np.meshgrid(xs, ys, zs, indexing="xy")
        voxel_centres = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()]).astype(float)
        dose_values = linear_grid.reshape(-1)

    plot_grid = clipped_grid
    plot_min = min_dose
    plot_max = max_dose
    if log_scale:
        plot_grid = np.log10(np.where(clipped_grid > 0, clipped_grid, min_dose))
        plot_min = np.log10(min_dose)
        plot_max = np.log10(max_dose)
        bar_title = "Log10 Dose (µSv/h)"
    else:
        bar_title = "Dose (µSv/h)"

    if voxel_centres is not None and dose_values is not None and stl_meshes:
        _attach_mesh_statistics(
            stl_meshes, voxel_centres, dose_values, voxel_volume, (dx, dy, dz)
        )

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

    vol = Volume(plot_grid, spacing=(dx, dy, dz), origin=(xs[0], ys[0], zs[0]))
    vol.cmap(cmap_name, vmin=plot_min, vmax=plot_max)
    vol.add_scalarbar(title=bar_title, size=(300, 900), font_size=36)
    try:
        vol._mcnp_dose_metadata = {  # type: ignore[attr-defined]
            "log_scale": log_scale,
            "conversion_factor": conversion_factor,
            "dose_quantile": dose_quantile,
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
    return vol, stl_meshes, cmap_name, plot_min, plot_max

def export_gltf(renwin):

    print("exporting scene.gltf")

    exporter = vtk.vtkGLTFExporter()
    exporter.SetFileName("scene.gltf")
    exporter.SetInput(renwin)
    exporter.SaveNormalOn()
    exporter.InlineDataOn()
    exporter.Update()

    exporter.Write()
    print("exported scene.gltf")
    
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
                stats_parts: list[str] = []
                stats = mesh_metadata.get("dose_statistics")
                if isinstance(stats, dict):
                    def _format_number(val: Any) -> float | None:
                        try:
                            number = float(val)
                        except (TypeError, ValueError):
                            return None
                        if not np.isfinite(number):
                            return None
                        return number

                    mean_val = _format_number(stats.get("mean_dose_rate"))
                    if mean_val is not None:
                        stats_parts.append(f"Mean dose: {mean_val:.3g} µSv/h")
                    voxel_count = stats.get("voxel_count")
                    if isinstance(voxel_count, (int, np.integer)) and voxel_count > 0:
                        stats_parts.append(f"Voxels: {int(voxel_count)}")
                    total_mass = _format_number(stats.get("total_mass_g"))
                    if total_mass is not None and total_mass > 0:
                        stats_parts.append(f"Mass: {total_mass:.3g} g")

                metadata_lines: list[str] = []
                if info_parts:
                    metadata_lines.append(" | ".join(info_parts))
                if stats_parts:
                    metadata_lines.append(" | ".join(stats_parts))
                if metadata_lines:
                    metadata_text = "\n".join(metadata_lines)

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
            slice_annotation = Text2D("", pos="top-right", bg="w", alpha=0.5)
            plt.add(slice_annotation)
        if hasattr(plt, "add_callback"):
            def _probe(evt: Any) -> None:
                annotation.text(_format_probe_text(evt))
                if hasattr(plt, "render"):
                    plt.render()

            plt.add_callback("MouseMove", _probe)
        else:
            slice_annotation = None

        volume_origin = (0.0, 0.0, 0.0)
        volume_spacing = (1.0, 1.0, 1.0)

        def _extract_volume_vector(attr: str, default: tuple[float, float, float]) -> tuple[float, float, float]:
            getter = getattr(vol, attr, None)
            values: Any
            if callable(getter):
                try:
                    values = getter()
                except Exception:  # pragma: no cover - vedo API variations
                    values = None
            else:
                values = getter
            try:
                arr = np.asarray(values, dtype=float)
            except Exception:
                return default
            if arr.size < 3 or not np.all(np.isfinite(arr[:3])):
                return default
            return float(arr[0]), float(arr[1]), float(arr[2])

        volume_origin = _extract_volume_vector("origin", volume_origin)
        volume_spacing = _extract_volume_vector("spacing", volume_spacing)

        axis_titles = [
            axes.get("xTitle", "X") if isinstance(axes, dict) else "X",
            axes.get("yTitle", "Y") if isinstance(axes, dict) else "Y",
            axes.get("zTitle", "Z") if isinstance(axes, dict) else "Z",
        ]

        def _axis_prefix(title: Any, fallback: str) -> str:
            if not isinstance(title, str):
                return fallback
            clean = title.strip()
            if not clean:
                return fallback
            if "(" in clean:
                clean = clean.split("(", 1)[0].strip()
            return clean or fallback

        def _axis_unit(title: Any) -> str:
            if not isinstance(title, str):
                return ""
            match = re.search(r"\(([^)]+)\)", title)
            if match:
                unit = match.group(1).strip()
                if unit:
                    return f" {unit}"
            return ""

        axis_prefixes = [
            _axis_prefix(axis_titles[0], "X"),
            _axis_prefix(axis_titles[1], "Y"),
            _axis_prefix(axis_titles[2], "Z"),
        ]
        axis_units = [
            _axis_unit(axis_titles[0]),
            _axis_unit(axis_titles[1]),
            _axis_unit(axis_titles[2]),
        ]

        def _sample_dose_at_point(point: tuple[float, float, float]) -> float | None:
            """Return the dose value at *point* or ``None`` if unavailable."""

            if point_factory is None:
                return None
            try:
                probed = point_factory(point).probe(vol)
            except Exception:
                return None
            try:
                values = np.asarray(probed.pointdata[0]).ravel()
            except Exception:
                values = np.asarray([])
            if values.size == 0:
                return None
            try:
                scalar_val = float(values[0])
            except (TypeError, ValueError):
                return None
            if not np.isfinite(scalar_val):
                return None
            log_scale = bool(metadata.get("log_scale", False))
            if log_scale:
                try:
                    dose_value = float(np.power(10.0, scalar_val))
                except OverflowError:
                    dose_value = float("inf")
            else:
                dose_value = scalar_val
            if not math.isfinite(dose_value):
                return None
            return dose_value

        def _format_slice_text() -> str:
            axis_parts: list[str] = []
            coord_values: list[float | None] = []
            sliders_iter = [
                getattr(plt, "xslider", None),
                getattr(plt, "yslider", None),
                getattr(plt, "zslider", None),
            ]
            for prefix, unit, slider, origin_val, spacing_val in zip(
                axis_prefixes,
                axis_units,
                sliders_iter,
                volume_origin,
                volume_spacing,
            ):
                slider_val = getattr(slider, "value", None)
                try:
                    slider_float = float(slider_val)
                except (TypeError, ValueError):
                    slider_float = None
                if slider_float is None or not math.isfinite(slider_float):
                    axis_parts.append(f"{prefix}: n/a")
                    coord_values.append(None)
                    continue
                coord = origin_val + slider_float * spacing_val
                if not math.isfinite(coord):
                    axis_parts.append(f"{prefix}: n/a")
                    coord_values.append(None)
                else:
                    axis_parts.append(f"{prefix}: {coord:.3g}{unit}")
                    coord_values.append(coord)

            text_parts = list(axis_parts)
            if len(coord_values) == 3 and all(coord is not None for coord in coord_values):
                point = (
                    float(coord_values[0]),
                    float(coord_values[1]),
                    float(coord_values[2]),
                )
                dose_value = _sample_dose_at_point(point)
                if dose_value is not None:
                    text_parts.append(f"Dose: {dose_value:.3g} µSv/h")

            return "Slice @ " + " | ".join(text_parts)

        def _update_slice_text() -> None:
            if slice_annotation is None:
                return
            slice_annotation.text(_format_slice_text())

        sliders = [getattr(plt, name, None) for name in ("xslider", "yslider", "zslider")]

        if slice_annotation is not None:
            _update_slice_text()

        def _slider_callback(_obj: Any = None, _evt: Any = None) -> None:
            _update_slice_text()
            if hasattr(plt, "render"):
                plt.render()

        for slider in sliders:
            if slider is not None and hasattr(slider, "AddObserver"):
                try:
                    slider.AddObserver("InteractionEvent", _slider_callback)
                    slider.AddObserver("EndInteractionEvent", _slider_callback)
                except Exception:  # pragma: no cover - best effort for slider API
                    continue

        renwin = None
        try:
            if hasattr(plt, "renderers"):
                renderers = getattr(plt, "renderers")
                if renderers:
                    renderer = renderers[0]
                    if hasattr(renderer, "GetRenderWindow"):
                        renwin = renderer.GetRenderWindow()
            elif hasattr(plt, "GetRenderWindow"):
                renwin = plt.GetRenderWindow()
        except Exception:  # pragma: no cover - renderer API differences
            renwin = None
        if renwin is not None:
            export_gltf(renwin)

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
