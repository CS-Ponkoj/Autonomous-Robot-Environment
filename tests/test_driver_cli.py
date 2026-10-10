"""Driver selection without opening a window, plus the autonomous speed display."""

import pygame
import pytest

from robot_env import app, config as C
from robot_env.baseline import BaselineDriver
from robot_env.manual import ManualInput


@pytest.fixture
def launched(monkeypatch):
    calls = []

    class CaptureApp:
        def __init__(self, *args, **kwargs):
            calls.append((args, kwargs))

        def run(self):
            return {"status": "running"}

    monkeypatch.setattr(app, "App", CaptureApp)
    return calls


@pytest.mark.parametrize("args", [[], ["--driver", "manual"]])
def test_manual_remains_default(args, launched):
    assert app.main(args) == {"status": "running"}
    assert launched[0][1]["driver"] is None


@pytest.mark.parametrize("level", [1, 2, 5])
def test_baseline_uses_selected_speed_and_preserves_options(level, launched):
    app.main(["--driver", "baseline", "--speed-level", str(level),
              "--seed", "1003", "--cats", "2", "--cat-seed", "16", "--view", "3"])
    args, kwargs = launched[0]
    driver = kwargs["driver"]
    assert isinstance(driver, BaselineDriver)
    assert (driver.v_max, driver.w_max) == C.SPEED_LEVELS[level - 1]
    assert args[0] == 1003 and args[4] == 3
    assert kwargs["speed_level"] == level - 1
    assert kwargs["cats"] == 2 and kwargs["cat_seed"] == 16


def test_baseline_default_speed(launched):
    app.main(["--driver", "baseline"])
    driver = launched[0][1]["driver"]
    assert (driver.v_max, driver.w_max) == C.SPEED_LEVELS[C.DEFAULT_SPEED_LEVEL]


def test_unknown_driver_rejected_before_opening_window(launched):
    with pytest.raises(SystemExit) as error:
        app.main(["--driver", "unknown"])
    assert error.value.code == 2 and not launched


def test_baseline_speed_display_ignores_manual_speed_inputs():
    viewer = app.App.__new__(app.App)
    viewer.driver = BaselineDriver(C.SPEED_LEVELS[0])
    viewer.input = ManualInput(0)
    before = viewer.speed_level_text()
    viewer.input.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_PLUS))
    viewer.input.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_LSHIFT))
    assert viewer.input.level == 1 and viewer.input.boosted
    assert viewer.speed_level_text() == before
    assert before == "baseline speed cap: 0.20 m/s, 0.8 rad/s (fixed)"
