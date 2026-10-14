"""The on-screen Manual / Autonomous switch (and the M key): who drives, changed mid-episode."""

import pygame
import pytest

from robot_env import config as C
from robot_env.app import App
from robot_env.baseline import BaselineDriver
from robot_env.manual import ManualDriver


def _click(app, pos):
    pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=pos))
    pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONUP, button=1, pos=pos))


@pytest.mark.gui
def test_buttons_and_m_key_switch_who_drives():
    app = App(1003, screenshot=None, frames=1)
    try:
        assert app.drive_mode == "manual" and isinstance(app.driver, ManualDriver)
        app.draw()  # places the buttons beside Options
        _click(app, app.drive_switch.buttons["autonomous"].center)
        app.handle_events()
        assert app.drive_mode == "autonomous" and isinstance(app.driver, BaselineDriver)
        assert (app.driver.v_max, app.driver.w_max) == C.SPEED_LEVELS[app.input.level]
        assert not app.menu.open  # the switch is not the Options button
        _click(app, app.drive_switch.buttons["manual"].center)
        app.handle_events()
        assert app.drive_mode == "manual" and isinstance(app.driver, ManualDriver)
        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_m, mod=0, unicode="m", scancode=0))
        app.handle_events()
        assert app.drive_mode == "autonomous"
        app.set_drive_mode("autonomous")  # choosing the mode in use changes nothing
        assert app.drive_mode == "autonomous"
        with pytest.raises(ValueError):
            app.set_drive_mode("model")
    finally:
        app._close_all()


@pytest.mark.gui
def test_autonomous_drives_toward_the_goal_and_manual_takes_back_and_stops():
    """Switched to autonomous mid-episode the baseline makes progress without keys; switched
    back to manual with no key held, the robot stops."""
    switch_back = 12.0

    def policy(app):
        t = app.system.time
        if t < 0.1 or switch_back <= t < switch_back + 0.1:
            return {pygame.K_m}
        return set()

    app = App(1003, screenshot=None, frames=1000, cats=0, key_policy=policy)
    start = app.system.observe().goal_distance
    seen = {}

    def watch(real_dt, original=app.simulate):
        original(real_dt)
        if app.drive_mode == "autonomous":
            seen["auto_distance"] = app.system.observe().goal_distance
    app.simulate = watch
    app.run()
    assert app.drive_mode == "manual"
    assert app.system.collisions == 0
    assert seen["auto_distance"] < start - 0.5 or app.episode.status == "success"
    assert abs(app.system.sim.true_velocity()[0]) < 0.02  # stopped once manual with no keys
