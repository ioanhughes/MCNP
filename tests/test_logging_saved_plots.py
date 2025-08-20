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
