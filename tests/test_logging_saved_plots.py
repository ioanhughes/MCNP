import logging
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from GUI import WidgetLoggerHandler

class DummyListbox:
    def __init__(self):
        self.items = []
    def insert(self, index, value):
        self.items.append(value)
    def after(self, delay, callback):
        callback()


def test_widget_logger_captures_saved_plot_path(tmp_path):
    listbox = DummyListbox()
    handler = WidgetLoggerHandler(None, listbox, None)
    handler.setFormatter(logging.Formatter("%(message)s"))
    root_logger = logging.getLogger()
    old_handlers = root_logger.handlers[:]
    root_logger.handlers = [handler]
    root_logger.setLevel(logging.INFO)
    try:
        logging.getLogger("he3_plotter.analysis").info(
            f"Saved: {tmp_path/'plot.pdf'}"
        )
    finally:
        root_logger.handlers = old_handlers
    assert listbox.items == [str(tmp_path/'plot.pdf')]


def test_widget_logger_thread_safe(tmp_path):
    """Logging from a background thread should still update the listbox."""

    listbox = DummyListbox()
    handler = WidgetLoggerHandler(None, listbox, None)
    handler.setFormatter(logging.Formatter("%(message)s"))
    root_logger = logging.getLogger()
    old_handlers = root_logger.handlers[:]
    root_logger.handlers = [handler]
    root_logger.setLevel(logging.INFO)
    path = tmp_path / "plot.pdf"

    def log_message():
        logging.getLogger("he3_plotter.analysis").info(f"Saved: {path}")

    try:
        import threading

        t = threading.Thread(target=log_message)
        t.start()
        t.join()
    finally:
        root_logger.handlers = old_handlers

    assert listbox.items == [str(path)]
