import logging
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
from mcnp.views.gui import WidgetLoggerHandler
from mcnp.he3_plotter.plots import plot_efficiency_and_rates
import pandas as pd

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
        logging.getLogger("mcnp.he3_plotter.analysis").info(
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
        logging.getLogger("mcnp.he3_plotter.analysis").info(f"Saved: {path}")

    try:
        import threading

        t = threading.Thread(target=log_message)
        t.start()
        t.join()
    finally:
        root_logger.handlers = old_handlers

    assert listbox.items == [str(path)]


def test_plot_efficiency_and_rates_logs_paths(tmp_path, caplog):
    df = pd.DataFrame(
        {
            "energy": [1.0],
            "rate_incident": [1.0],
            "rate_detected": [0.5],
            "rate_incident_err": [0.1],
            "rate_detected_err": [0.05],
            "efficiency": [0.5],
            "efficiency_err": [0.01],
        }
    )
    dummy = tmp_path / "test.o"
    dummy.write_text("dummy")
    with caplog.at_level(logging.INFO, logger="mcnp.he3_plotter.plots"):
        plot_efficiency_and_rates(df, dummy)
    messages = [rec.message for rec in caplog.records if "Saved:" in rec.message]
    saved_paths = [m.split("Saved:", 1)[1].strip() for m in messages]
    assert len(saved_paths) == 2
    for p in saved_paths:
        assert Path(p).exists()


def test_plot_efficiency_and_rates_multiple_surfaces(tmp_path, caplog):
    df = pd.DataFrame(
        {
            "energy": [1.0, 1.0],
            "rate_incident": [1.0, 2.0],
            "rate_detected": [0.5, 1.0],
            "rate_incident_err": [0.1, 0.2],
            "rate_detected_err": [0.05, 0.1],
            "efficiency": [0.5, 0.5],
            "efficiency_err": [0.01, 0.02],
            "surface": [1, 2],
        }
    )
    dummy = tmp_path / "test_multi.o"
    dummy.write_text("dummy")
    with caplog.at_level(logging.INFO, logger="mcnp.he3_plotter.plots"):
        plot_efficiency_and_rates(df, dummy)
    messages = [rec.message for rec in caplog.records if "Saved:" in rec.message]
    saved_paths = [m.split("Saved:", 1)[1].strip() for m in messages]
    assert len(saved_paths) == 2
    for p in saved_paths:
        assert Path(p).exists()
