import os
from datetime import datetime
from tkinter import Tk
from tkinter.filedialog import askopenfilename, askdirectory
import atexit

from .config import config

_hidden_root = None


def _get_hidden_root():
    global _hidden_root
    if _hidden_root is None:
        _hidden_root = Tk()
        _hidden_root.withdraw()
        atexit.register(_hidden_root.destroy)
    return _hidden_root


def select_file(title: str = "Select a file"):
    """Prompt the user to select a single file."""

    _get_hidden_root()
    return askopenfilename(title=title)


def select_folder(title: str = "Select a folder"):
    """Prompt the user to select a folder."""

    _get_hidden_root()
    return askdirectory(title=title)


def get_output_path(base_path, filename_prefix, descriptor, extension=None, subfolder="plots"):
    """Return a filesystem path for saving output files."""

    if extension is None:
        extension = config.plot_extension
    output_dir = os.path.join(base_path, subfolder)
    os.makedirs(output_dir, exist_ok=True)
    date_str = datetime.now().strftime("%Y-%m-%d")
    tag = f" {config.filename_tag.strip()}" if config.filename_tag.strip() else ""
    filename = f"{filename_prefix} {descriptor}{tag} {date_str}.{extension}"
    return os.path.join(output_dir, filename)
