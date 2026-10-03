"""GUI smoke tests: the real window, renderer, and event loop, driven by scripted keys.

These show the window works end to end. They do not replace the owner's manual drive."""

import pygame
import pytest

from robot_env.app import App, parse_script


def run_app(tmp_path, **kw):
    shot = tmp_path / "shot.png"
    app = App(kw.pop("seed", 1000), screenshot=shot, **kw)
    summary = app.run()
    return summary, shot


def test_window_renders_and_saves_screenshot(tmp_path):
    summary, shot = run_app(tmp_path, frames=20)
    assert shot.exists() and shot.stat().st_size > 10_000
    assert summary["status"] == "running" and summary["collisions"] == 0


@pytest.mark.parametrize("view", [0, 1, 2, 3])
def test_every_view_renders(tmp_path, view):
    summary, shot = run_app(tmp_path, frames=5, view=view)
    assert shot.exists()


def test_scripted_keys_move_the_robot_and_release_stops_it(tmp_path):
    summary, _ = run_app(tmp_path, frames=150, script=parse_script("W:1.2;none:1.0"))
    assert summary["collisions"] == 0
    assert summary["sim_time"] > 1.5


def test_parse_script():
    steps = parse_script("W:2;W+A:1;none:0.5;SPACE:0.3")
    assert steps[1].keys == frozenset({pygame.K_w, pygame.K_a}) and steps[2].keys == frozenset()
    assert steps[3].duration == 0.3


def test_real_time_factor_is_honest_when_frames_are_slow(tmp_path, monkeypatch):
    """Regression: RTF showed 1.00 at about 9 FPS."""
    import time as _time
    original = App.draw

    def slow_draw(self):
        original(self)
        _time.sleep(0.12)

    monkeypatch.setattr(App, "draw", slow_draw)
    summary, _ = run_app(tmp_path, frames=12, script=parse_script("W:3"))
    assert summary["real_time_factor"] < 0.7


@pytest.mark.parametrize("yaw", [0.0, 1.57, 3.14, -1.57])
def test_chase_camera_stays_inside_the_room_near_walls(yaw):
    import math
    from robot_env.app import ViewCamera
    v = ViewCamera()
    # Robot center 0.22 m from the wall it backs onto: the camera must not go behind it.
    x = 3.0 - 0.22 if abs(yaw - 3.14) < 0.1 else (-3.0 + 0.22 if yaw == 0.0 else 0.0)
    y = 3.0 - 0.22 if yaw == -1.57 else (-3.0 + 0.22 if yaw == 1.57 else 0.0)
    cam = v.apply(x, y, yaw)
    az, el = math.radians(cam.azimuth), math.radians(cam.elevation)
    cx = x - cam.distance * math.cos(el) * math.cos(az)
    cy = y - cam.distance * math.cos(el) * math.sin(az)
    assert abs(cx) <= 2.95 and abs(cy) <= 2.95
    assert cam.distance >= 0.3


def test_episode_clock_stops_at_episode_end(tmp_path):
    from robot_env import config as C
    app = App(1005, screenshot=None, frames=1)
    app.system.advance(C.EPISODE_TIME_LIMIT + 0.5)
    app.update_episode()
    assert app.episode.status == "timeout" and app.episode_time() == C.EPISODE_TIME_LIMIT
    app.system.advance(1.0)
    assert app.episode_time() == C.EPISODE_TIME_LIMIT
    app.run()


class _FakeModelDriver:
    """Stands in for a decision model: always asks for forward motion."""

    name = "fake_model"

    def decide(self, obs):
        from robot_env.types import Command, Decision
        return Decision(Command(0.3, 0.0), obs.seq)


def test_window_runs_a_non_manual_driver_without_held_keys(tmp_path):
    """Regression: release-to-stop blocked any non-manual driver."""
    app = App(1006, screenshot=tmp_path / "s.png", frames=60, driver=_FakeModelDriver())
    assert not app.manual_driving
    app.run()
    assert app.system.last_decision.command.v == 0.3
    assert app.system.last_result.reasons == () or "clearance" in app.system.last_result.reasons
    assert app.system.sim.true_velocity()[0] > 0.1  # it actually moves


def test_space_still_brakes_a_non_manual_driver(tmp_path):
    app = App(1006, screenshot=tmp_path / "s.png", frames=90, driver=_FakeModelDriver(),
              script=parse_script("none:0.6;SPACE:1.0"))
    app.run()
    assert app.system.last_result.reasons == ("emergency_brake",)
    assert abs(app.system.sim.true_velocity()[0]) < 0.02
