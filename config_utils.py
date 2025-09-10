import json
from pathlib import Path

# Paths used for storing configuration
PROJECT_SETTINGS_PATH = Path(__file__).resolve().parent / "config.json"
HOME_SETTINGS_PATH = Path.home() / ".mcnp_tools_settings.json"


def load_settings() -> dict:
    """Load settings from the project or user configuration file.

    Settings are read with the following precedence:

    1. Project-local ``config.json``
    2. Legacy user config ``~/.mcnp_tools_settings.json``
    """
    for path in (PROJECT_SETTINGS_PATH, HOME_SETTINGS_PATH):
        try:
            if path.exists():
                with open(path, "r") as f:
                    return json.load(f)
        except Exception:
            # Ignore malformed files and fall back to next option
            pass
    return {}


def save_settings(data: dict) -> None:
    """Persist ``data`` to the project ``config.json`` file.

    Existing settings are merged so that only provided keys are updated.
    """
    try:
        if PROJECT_SETTINGS_PATH.exists():
            with open(PROJECT_SETTINGS_PATH, "r") as f:
                existing = json.load(f)
        else:
            existing = {}
        existing.update(data)
        with open(PROJECT_SETTINGS_PATH, "w") as f:
            json.dump(existing, f)
    except Exception:
        # Silently ignore I/O errors to avoid disrupting the caller
        pass
