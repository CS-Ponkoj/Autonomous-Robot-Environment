"""QA tool behavior that needs no window."""

import types

import pytest


def test_fps_renderer_is_read_from_the_apps_context(monkeypatch):
    """The renderer string is queried after the app's rendering context renders, and missing
    PyOpenGL is reported honestly rather than guessed."""
    import sys

    import tools.fps_protocol as fp
    calls = []
    app = types.SimpleNamespace(
        system=types.SimpleNamespace(sim=types.SimpleNamespace(data=None)),
        renderer=types.SimpleNamespace(update_scene=lambda data: calls.append("scene"),
                                       render=lambda: calls.append("render")))
    gl = types.SimpleNamespace(GL_RENDERER=7, glGetString=lambda k: b"Mock GPU/PCIe" if calls == ["scene", "render"] else b"")
    monkeypatch.setitem(sys.modules, "OpenGL", types.SimpleNamespace(GL=gl))
    monkeypatch.setitem(sys.modules, "OpenGL.GL", gl)
    assert fp.renderer_string(app) == "Mock GPU/PCIe"
    monkeypatch.setitem(sys.modules, "OpenGL", None)  # import fails
    calls.clear()
    assert fp.renderer_string(app) == "unavailable (PyOpenGL not installed)"


def test_reference_path_ratio_is_documented_as_able_to_exceed_one():
    import inspect

    import tools.eval_baseline as ev
    src = inspect.getsource(ev.run_episode)
    assert "reference_path_ratio" in src and "path_efficiency" not in src
    assert ev.SCHEMA == 2


if __name__ == "__main__":
    pytest.main([__file__])
