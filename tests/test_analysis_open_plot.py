import logging
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
import mcnp.views.analysis as analysis_module


class DummyListbox:
    def __init__(self, path: Path):
        self._path = path

    def curselection(self):
        return (0,)

    def get(self, index):
        assert index == 0
        return str(self._path)


class DummyApp:
    def __init__(self):
        self.logged: list[tuple[str, int]] = []

    def log(self, message: str, level: int = logging.INFO) -> None:
        self.logged.append((message, level))


def _make_view(path: Path):
    view = object.__new__(analysis_module.AnalysisView)
    view.app = DummyApp()
    view.plot_listbox = DummyListbox(path)
    return view


def test_open_selected_plot_linux_non_blocking(monkeypatch, tmp_path):
    plot_path = tmp_path / "plot.pdf"
    plot_path.write_text("dummy")
    view = _make_view(plot_path)

    calls = []

    def fake_popen(cmd, stdout=None, stderr=None, start_new_session=False):
        calls.append((cmd, stdout, stderr, start_new_session))

        class DummyProcess:
            pass

        return DummyProcess()

    monkeypatch.setattr(analysis_module.sys, "platform", "linux")
    monkeypatch.setattr(analysis_module.subprocess, "Popen", fake_popen)

    view.open_selected_plot(None)

    assert calls
    cmd, stdout, stderr, start_new_session = calls[0]
    assert cmd == ["xdg-open", str(plot_path)]
    assert stdout is analysis_module.subprocess.DEVNULL
    assert stderr is analysis_module.subprocess.DEVNULL
    assert start_new_session is True


def test_open_selected_plot_macos_non_blocking(monkeypatch, tmp_path):
    plot_path = tmp_path / "plot.pdf"
    plot_path.write_text("dummy")
    view = _make_view(plot_path)

    calls = []

    def fake_popen(cmd, stdout=None, stderr=None, start_new_session=False):
        calls.append((cmd, stdout, stderr, start_new_session))

        class DummyProcess:
            pass

        return DummyProcess()

    monkeypatch.setattr(analysis_module.sys, "platform", "darwin")
    monkeypatch.setattr(analysis_module.subprocess, "Popen", fake_popen)

    view.open_selected_plot(None)

    assert calls
    cmd, stdout, stderr, start_new_session = calls[0]
    assert cmd == ["open", str(plot_path)]
    assert stdout is analysis_module.subprocess.DEVNULL
    assert stderr is analysis_module.subprocess.DEVNULL
    assert start_new_session is True


def test_open_selected_plot_windows_uses_startfile(monkeypatch, tmp_path):
    plot_path = tmp_path / "plot.pdf"
    plot_path.write_text("dummy")
    view = _make_view(plot_path)

    called = []

    def fake_startfile(path):
        called.append(path)

    monkeypatch.setattr(analysis_module.sys, "platform", "win32")
    monkeypatch.setattr(analysis_module.os, "startfile", fake_startfile, raising=False)

    view.open_selected_plot(None)

    assert called == [str(plot_path)]
