"""Manual input with synthetic pygame events (no window needed)."""

import pygame

from robot_env import config as C
from robot_env.manual import ManualInput
from robot_env.types import Command


def key(t, k):
    return pygame.event.Event(t, key=k, mod=0, unicode="", scancode=0)


def press(m, *keys):
    for k in keys:
        m.handle_event(key(pygame.KEYDOWN, k))


def release(m, *keys):
    for k in keys:
        m.handle_event(key(pygame.KEYUP, k))


NORMAL = C.SPEED_LEVELS[C.DEFAULT_SPEED_LEVEL]
FASTEST = C.SPEED_LEVELS[-1]


def test_keys_map_to_commands():
    m = ManualInput()
    press(m, pygame.K_w)
    assert m.command() == Command(NORMAL[0], 0.0) and m.drive_input_held
    press(m, pygame.K_a)
    assert m.command() == Command(NORMAL[0], NORMAL[1])
    press(m, pygame.K_LSHIFT)
    assert m.command() == Command(FASTEST[0], FASTEST[1])
    release(m, pygame.K_w, pygame.K_a, pygame.K_LSHIFT)
    press(m, pygame.K_DOWN, pygame.K_RIGHT)
    assert m.command() == Command(-NORMAL[0] * C.MANUAL_REVERSE_FACTOR, -NORMAL[1])


def test_speed_levels_are_the_single_source_of_the_limits():
    assert list(C.SPEED_LEVELS) == sorted(C.SPEED_LEVELS)  # slowest first, contiguous indices
    assert C.MAX_LINEAR_SPEED == max(v for v, _ in C.SPEED_LEVELS)
    assert all(0 < v <= C.MAX_LINEAR_SPEED and 0 < w <= C.MAX_ANGULAR_SPEED for v, w in C.SPEED_LEVELS)
    assert 0 <= C.DEFAULT_SPEED_LEVEL < len(C.SPEED_LEVELS)
    assert C.SPEED_LEVELS[C.DEFAULT_SPEED_LEVEL] == (0.30, 1.0)  # normal driving is unchanged


def test_wheel_targets_never_saturate_at_the_fastest_command():
    from robot_env.sim import RobotSim
    sim = RobotSim(include_obstacles=False)
    limit = sim.model.actuator_ctrlrange[:, 1].min()
    for v, w in list(C.SPEED_LEVELS) + [(C.MAX_LINEAR_SPEED, C.MAX_ANGULAR_SPEED)]:
        wheel = (abs(v) + abs(w) * C.WHEEL_SEPARATION / 2) / C.WHEEL_RADIUS
        assert wheel <= limit, (v, w, wheel, limit)
    sim.close()


def test_level_keys_step_once_per_press_and_clamp():
    m = ManualInput()
    for k in (pygame.K_EQUALS, pygame.K_KP_PLUS, pygame.K_PLUS):
        press(m, k)
        release(m, k)
    assert m.level == len(C.SPEED_LEVELS) - 1  # clamped at the top
    for k in (pygame.K_MINUS, pygame.K_KP_MINUS, pygame.K_MINUS, pygame.K_MINUS, pygame.K_MINUS):
        press(m, k)
        release(m, k)
    assert m.level == 0  # clamped at the bottom
    press(m, pygame.K_EQUALS)  # one KEYDOWN, held: exactly one step (no key repeat)
    assert m.level == 1


def test_level_keys_are_not_drive_input_and_never_touch_the_brake():
    m = ManualInput()
    press(m, pygame.K_EQUALS)
    assert not m.drive_input_held and m.command() == Command()  # never moves the robot alone
    press(m, pygame.K_SPACE)
    release(m, pygame.K_SPACE, pygame.K_EQUALS)
    press(m, pygame.K_EQUALS, pygame.K_MINUS, pygame.K_KP_PLUS)  # level keys while braked
    assert m.emergency_brake and not m.drive_input_held
    press(m, pygame.K_w)  # a real drive press still releases it
    assert not m.emergency_brake
    m.handle_event(pygame.event.Event(pygame.WINDOWFOCUSLOST))
    press(m, pygame.K_MINUS)
    assert m.pending_release == {pygame.K_w}  # level keys never become pending inputs
    m.handle_event(pygame.event.Event(pygame.WINDOWFOCUSGAINED))
    release(m, pygame.K_w, pygame.K_MINUS, pygame.K_EQUALS, pygame.K_KP_PLUS)


def test_shift_boost_with_both_shift_keys_and_a_base_change_while_boosted():
    m = ManualInput()
    press(m, pygame.K_w)
    press(m, pygame.K_LSHIFT, pygame.K_RSHIFT)
    assert m.command().v == FASTEST[0] and m.effective_level == len(C.SPEED_LEVELS) - 1
    release(m, pygame.K_LSHIFT)  # the other Shift is still held: boost stays
    assert m.command().v == FASTEST[0]
    press(m, pygame.K_MINUS)  # base level changes, effective target does not
    assert m.level == C.DEFAULT_SPEED_LEVEL - 1 and m.command().v == FASTEST[0]
    release(m, pygame.K_RSHIFT)  # back to the (new) selected level
    assert m.command().v == C.SPEED_LEVELS[C.DEFAULT_SPEED_LEVEL - 1][0]


def test_reverse_is_half_the_effective_level_and_mouse_uses_the_effective_level():
    m = ManualInput(len(C.SPEED_LEVELS) - 1)
    press(m, pygame.K_s)
    assert m.command().v == -FASTEST[0] * C.MANUAL_REVERSE_FACTOR
    release(m, pygame.K_s)
    m.level = 0
    m.handle_event(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=3, pos=(400, 400)))
    m.handle_event(pygame.event.Event(pygame.MOUSEMOTION, pos=(400 - 150, 400 - 150), rel=(0, 0), buttons=(0, 0, 1)))
    assert m.command() == Command(C.SPEED_LEVELS[0][0], C.SPEED_LEVELS[0][1])
    press(m, pygame.K_LSHIFT)
    assert m.command() == Command(FASTEST[0], FASTEST[1])


def test_invalid_starting_level_is_rejected():
    import pytest
    with pytest.raises(ValueError):
        ManualInput(len(C.SPEED_LEVELS))


def test_release_means_not_held():
    m = ManualInput()
    press(m, pygame.K_w)
    release(m, pygame.K_w)
    assert not m.drive_input_held and m.command() == Command()


def test_opposite_keys_cancel():
    m = ManualInput()
    press(m, pygame.K_w, pygame.K_s)
    assert m.command().v == 0.0


def test_shift_alone_is_not_drive_input():
    m = ManualInput()
    press(m, pygame.K_LSHIFT)
    assert not m.drive_input_held


def test_space_brake_latches_until_release_and_new_press():
    m = ManualInput()
    press(m, pygame.K_w)
    press(m, pygame.K_SPACE)
    assert m.emergency_brake
    press(m, pygame.K_a)  # still holding W: does not clear the brake
    assert m.emergency_brake
    release(m, pygame.K_w, pygame.K_a, pygame.K_SPACE)
    assert m.emergency_brake  # released, still braked
    press(m, pygame.K_w)  # a new press after release clears it
    assert not m.emergency_brake and m.drive_input_held


def test_focus_loss_releases_everything():
    m = ManualInput()
    press(m, pygame.K_w)
    m.handle_event(pygame.event.Event(pygame.WINDOWFOCUSLOST))
    assert not m.focused and not m.drive_input_held
    m.handle_event(pygame.event.Event(pygame.WINDOWFOCUSGAINED))
    assert m.focused and not m.drive_input_held  # needs a new press to move again


def test_mouse_right_drag_drives():
    m = ManualInput()
    m.handle_event(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=3, pos=(400, 400)))
    m.handle_event(pygame.event.Event(pygame.MOUSEMOTION, pos=(400 - 75, 400 - 150), rel=(0, 0), buttons=(0, 0, 1)))
    c = m.command()
    assert m.drive_input_held
    assert abs(c.v - NORMAL[0]) < 1e-9  # full drag up = full forward
    assert abs(c.omega - 0.5 * NORMAL[1]) < 1e-9  # half drag left = half left turn
    m.handle_event(pygame.event.Event(pygame.MOUSEBUTTONUP, button=3, pos=(325, 250)))
    assert not m.drive_input_held


def test_left_mouse_does_not_drive():
    m = ManualInput()
    m.handle_event(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=(400, 400)))
    assert not m.drive_input_held


def test_brake_stays_on_while_space_is_held():
    """Regression: Space then W while Space is still held released the brake."""
    m = ManualInput()
    press(m, pygame.K_SPACE)
    press(m, pygame.K_w)
    assert m.emergency_brake
    release(m, pygame.K_w)
    press(m, pygame.K_w)
    assert m.emergency_brake  # Space still held
    release(m, pygame.K_w, pygame.K_SPACE)
    press(m, pygame.K_w)
    assert not m.emergency_brake


def test_space_tap_with_no_keys_then_drive_key_releases_brake():
    m = ManualInput()
    press(m, pygame.K_SPACE)
    release(m, pygame.K_SPACE)
    assert m.emergency_brake
    press(m, pygame.K_w)
    assert not m.emergency_brake


def test_manual_driver_uses_the_decide_plug():
    import numpy as np
    from robot_env import config as C
    from robot_env.manual import ManualDriver
    from robot_env.types import Observation
    obs = Observation(42, 0.0, np.zeros(C.LIDAR_RAYS), np.ones(C.LIDAR_RAYS, bool),
                      np.array(C.LIDAR_ANGLES), 0.0, (0.0, 0.0), 1.0, 0.0, False)
    m = ManualInput()
    d = ManualDriver(m)
    assert d.decide(obs) is None  # nothing held: no fresh command
    press(m, pygame.K_w)
    decision = d.decide(obs)
    assert decision.command == Command(NORMAL[0], 0.0) and decision.obs_seq == 42


def test_focus_loss_does_not_forget_a_held_space():
    """Regression: Space down, focus lost and regained, W down without Space up."""
    m = ManualInput()
    press(m, pygame.K_SPACE)
    m.handle_event(pygame.event.Event(pygame.WINDOWFOCUSLOST))
    m.handle_event(pygame.event.Event(pygame.WINDOWFOCUSGAINED))
    press(m, pygame.K_w)
    assert m.emergency_brake and m.space_held
    release(m, pygame.K_w, pygame.K_SPACE)  # a genuine Space release
    press(m, pygame.K_w)
    assert not m.emergency_brake


def test_forgotten_but_still_held_key_blocks_brake_release():
    """Regression: W down, Space tap, restart forgets W (still physically down),
    D down -> the brake must stay on until W is genuinely released."""
    m = ManualInput()
    press(m, pygame.K_w)
    press(m, pygame.K_SPACE)
    release(m, pygame.K_SPACE)
    m.release_all()  # what a restart (R / N) or focus loss does
    press(m, pygame.K_d)
    assert m.emergency_brake and "W" in m.brake_hint()
    release(m, pygame.K_d)
    release(m, pygame.K_w)  # the genuine release of W
    assert m.brake_hint() == "press a drive key to go"
    press(m, pygame.K_d)
    assert not m.emergency_brake


def test_forgotten_right_mouse_drag_blocks_brake_release():
    m = ManualInput()
    m.handle_event(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=3, pos=(400, 400)))
    press(m, pygame.K_SPACE)
    release(m, pygame.K_SPACE)
    m.handle_event(pygame.event.Event(pygame.WINDOWFOCUSLOST))
    m.handle_event(pygame.event.Event(pygame.WINDOWFOCUSGAINED))
    press(m, pygame.K_w)
    assert m.emergency_brake and "right mouse button" in m.brake_hint()
    release(m, pygame.K_w)
    m.handle_event(pygame.event.Event(pygame.MOUSEBUTTONUP, button=3, pos=(400, 400)))
    press(m, pygame.K_w)
    assert not m.emergency_brake


def test_pending_names_include_the_mouse_button():
    """Regression: the restart hint must name the input actually awaiting release."""
    m = ManualInput()
    m.handle_event(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=3, pos=(10, 10)))
    press(m, pygame.K_w)
    m.release_all()
    assert m.pending_names() == ["W", "right mouse button"]
