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


def test_keys_map_to_commands():
    m = ManualInput()
    press(m, pygame.K_w)
    assert m.command() == Command(C.MANUAL_NORMAL[0], 0.0) and m.drive_input_held
    press(m, pygame.K_a)
    assert m.command() == Command(C.MANUAL_NORMAL[0], C.MANUAL_NORMAL[1])
    press(m, pygame.K_LSHIFT)
    assert m.command() == Command(C.MANUAL_FAST[0], C.MANUAL_FAST[1])
    release(m, pygame.K_w, pygame.K_a, pygame.K_LSHIFT)
    press(m, pygame.K_DOWN, pygame.K_RIGHT)
    assert m.command() == Command(-C.MANUAL_NORMAL[0] * C.MANUAL_REVERSE_FACTOR, -C.MANUAL_NORMAL[1])


def test_fast_speeds_are_within_safety_limits():
    assert C.MANUAL_FAST[0] <= C.MAX_LINEAR_SPEED and C.MANUAL_FAST[1] <= C.MAX_ANGULAR_SPEED


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
    assert abs(c.v - C.MANUAL_NORMAL[0]) < 1e-9  # full drag up = full forward
    assert abs(c.omega - 0.5 * C.MANUAL_NORMAL[1]) < 1e-9  # half drag left = half left turn
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
    assert decision.command == Command(C.MANUAL_NORMAL[0], 0.0) and decision.obs_seq == 42
