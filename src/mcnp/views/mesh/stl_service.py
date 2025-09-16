from __future__ import annotations

"""Utilities for loading and caching STL meshes for the mesh tally view."""

import copy
import logging
import os
from typing import Any


LOGGER = logging.getLogger(__name__)


class StlMeshService:
    """Manage STL meshes, subdivision caching, and persistence helpers."""

    def __init__(self, vp_module: Any) -> None:
        self._vp = vp_module
        self._stl_folder: str | None = None
        self._stl_files: list[str] | None = None
        self._raw_meshes: list[Any] | None = None
        self._subdivision_cache: dict[int, list[Any]] = {}

    @property
    def available(self) -> bool:
        """Return ``True`` when the vedo module is available."""

        return getattr(self._vp, "vedo", None) is not None

    @property
    def stl_folder(self) -> str | None:
        """Return the folder containing the currently loaded STL files."""

        return self._stl_folder

    @stl_folder.setter
    def stl_folder(self, value: str | None) -> None:
        self._stl_folder = value

    @property
    def stl_files(self) -> list[str] | None:
        """Return the filenames for the loaded STL meshes."""

        return self._stl_files

    @property
    def has_meshes(self) -> bool:
        """Return ``True`` when STL meshes have been loaded."""

        return bool(self._raw_meshes)

    def clear(self) -> None:
        """Reset cached STL meshes and metadata."""

        self._raw_meshes = None
        self._subdivision_cache = {}
        self._stl_files = None
        self._stl_folder = None

    def read_folder(self, folderpath: str) -> tuple[list[Any], list[str]]:
        """Load meshes from *folderpath* without mutating internal state."""

        loader = getattr(self._vp, "load_stl_meshes", None)
        if loader is None or getattr(self._vp, "vedo", None) is None:
            raise RuntimeError("Vedo library not available")
        return loader(folderpath, 0)

    def update_meshes(
        self, folderpath: str, meshes: list[Any], stl_files: list[str]
    ) -> None:
        """Store meshes loaded from *folderpath* and prime the cache."""

        self._stl_folder = folderpath
        self._stl_files = list(stl_files)
        self._raw_meshes = list(meshes)
        self._subdivision_cache = {0: list(meshes)}

    def get_base_meshes(self) -> list[Any]:
        """Return the raw meshes suitable for immediate rendering."""

        if self._raw_meshes is None:
            return []
        base = self._subdivision_cache.get(0)
        if base is None:
            base = list(self._raw_meshes)
            self._subdivision_cache[0] = base
        return base

    def get_meshes_for_level(self, level: int) -> list[Any]:
        """Return meshes subdivided to *level*, caching results."""

        base_meshes = self._raw_meshes
        if base_meshes is None:
            return []

        try:
            level = int(level)
        except (TypeError, ValueError):
            level = 0
        if level < 0:
            level = 0

        cache = self._subdivision_cache
        if level in cache:
            return cache[level]

        meshes: list[Any] = []
        for idx, mesh in enumerate(base_meshes):
            clone = self._clone_mesh(idx, mesh, reuse_base=level == 0)
            if level > 0 and getattr(self._vp, "vedo", None) is not None:
                try:
                    clone = clone.triangulate().subdivide(level, method=1)
                except Exception:
                    try:
                        clone.triangulate()
                    except Exception:
                        pass
                    try:
                        clone.subdivide(level, method=1)
                    except Exception:
                        pass
            meshes.append(clone)

        cache[level] = meshes
        return meshes

    def save_to_folder(self, folderpath: str, level: int) -> int:
        """Persist subdivided meshes to *folderpath* and return the count."""

        if getattr(self._vp, "vedo", None) is None:
            raise RuntimeError("Vedo library not available")

        stl_files = self._stl_files or []
        meshes = self.get_meshes_for_level(level)
        if not meshes or not stl_files:
            raise ValueError("No STL files loaded")

        saved = 0
        for mesh, name in zip(meshes, stl_files):
            try:
                mesh.write(os.path.join(folderpath, name))
            except Exception as exc:  # pragma: no cover - write errors
                LOGGER.error("Failed to save STL %s: %s", name, exc)
                continue
            saved += 1
        return saved

    # ------------------------------------------------------------------
    def _clone_mesh(self, index: int, mesh: Any, *, reuse_base: bool) -> Any:
        """Return a copy of *mesh* suitable for further processing."""

        if reuse_base:
            return mesh

        metadata_attr = getattr(self._vp, "MESH_METADATA_ATTR", None)
        mesh_metadata = None
        if metadata_attr:
            mesh_metadata = getattr(mesh, metadata_attr, None)

        def _with_metadata(candidate: Any) -> Any:
            if (
                candidate is not mesh
                and metadata_attr
                and mesh_metadata is not None
                and candidate is not None
            ):
                try:
                    setattr(candidate, metadata_attr, mesh_metadata)
                except Exception:
                    pass
            return candidate

        clone_method = getattr(mesh, "clone", None)
        if callable(clone_method):
            try:
                return _with_metadata(clone_method())
            except Exception:
                pass

        copy_method = getattr(mesh, "copy", None)
        if callable(copy_method):
            for arg in ({}, {"deep": True}, {"deepcopy": True}):
                try:
                    return _with_metadata(copy_method(**arg))
                except TypeError:
                    continue
                except Exception:
                    break

        if self._stl_folder and self._stl_files and index < len(self._stl_files):
            vedo_mod = getattr(self._vp, "vedo", None)
            if vedo_mod is not None:
                path = os.path.join(self._stl_folder, self._stl_files[index])
                try:
                    mesh_obj = vedo_mod.Mesh(path).alpha(1).c("lightblue").wireframe(False)
                except Exception:
                    mesh_obj = None
                else:
                    return _with_metadata(mesh_obj)

        try:
            return _with_metadata(copy.deepcopy(mesh))
        except Exception:
            return mesh
