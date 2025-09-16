"""Helpers for persisting view configuration data."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict


@dataclass(slots=True)
class JsonConfigStore:
    """Simple JSON-backed store for persisting configuration dictionaries."""

    path: Path

    def load(self) -> Dict[str, Any]:
        """Return the JSON payload stored at :attr:`path` if it exists."""

        if not self.path.exists():
            return {}
        with open(self.path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    def merge(self, updates: Dict[str, Any]) -> Dict[str, Any]:
        """Merge ``updates`` into the existing JSON configuration."""

        data = self.load()
        data.update(updates)
        with open(self.path, "w", encoding="utf-8") as handle:
            json.dump(data, handle)
        return data
