"""Manual driving input. Event driven, so it can be tested with synthetic pygame events.

Keyboard: W/S or Up/Down drive, A/D or Left/Right turn, hold Shift for fast.
Mouse: hold the right button and drag (up = forward, sideways = turn).
Space: emergency brake. It latches: the robot stays stopped until Space and every
drive input have been released and a drive input is pressed again.
Losing window focus releases every input (the robot stops).
"""

from __future__ import annotations

import pygame

from . import config as C
from .types import Command, Decision, Observation

FORWARD_KEYS = {pygame.K_w, pygame.K_UP}
BACK_KEYS = {pygame.K_s, pygame.K_DOWN}
LEFT_KEYS = {pygame.K_a, pygame.K_LEFT}
RIGHT_KEYS = {pygame.K_d, pygame.K_RIGHT}
DRIVE_KEYS = FORWARD_KEYS | BACK_KEYS | LEFT_KEYS | RIGHT_KEYS
SHIFT_KEYS = {pygame.K_LSHIFT, pygame.K_RSHIFT}
DRAG_FULL_SCALE = 150.0  # pixels of drag for full speed


class ManualInput:
    def __init__(self) -> None:
        self.held: set[int] = set()
        self.drag_origin: tuple[int, int] | None = None
        self.drag_pos: tuple[int, int] | None = None
        self.emergency_brake = False
        self._brake_release_armed = False
        self.space_held = False
        self.focused = True

    def handle_event(self, event: pygame.event.Event) -> None:
        t = event.type
        if t == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                self.space_held = True
                self.emergency_brake = True
                self._brake_release_armed = False
            elif event.key in DRIVE_KEYS or event.key in SHIFT_KEYS:
                if event.key in DRIVE_KEYS and self.emergency_brake and self._brake_release_armed:
                    self.emergency_brake = False
                self.held.add(event.key)
        elif t == pygame.KEYUP:
            self.held.discard(event.key)
            if event.key == pygame.K_SPACE:
                self.space_held = False
        elif t == pygame.MOUSEBUTTONDOWN and event.button == 3:
            if self.emergency_brake and self._brake_release_armed:
                self.emergency_brake = False
            self.drag_origin = self.drag_pos = event.pos
        elif t == pygame.MOUSEMOTION and self.drag_origin is not None:
            self.drag_pos = event.pos
        elif t == pygame.MOUSEBUTTONUP and event.button == 3:
            self.drag_origin = self.drag_pos = None
        elif t == pygame.WINDOWFOCUSLOST:
            self.focused = False
            self.release_all()
        elif t == pygame.WINDOWFOCUSGAINED:
            self.focused = True
        if self.emergency_brake and not self.drive_input_held and not self.space_held:
            self._brake_release_armed = True

    def release_all(self) -> None:
        self.held.clear()
        self.space_held = False
        self.drag_origin = self.drag_pos = None

    @property
    def drive_input_held(self) -> bool:
        return bool(self.held & DRIVE_KEYS) or self.drag_origin is not None

    def command(self) -> Command:
        """The requested command from the inputs held right now."""
        forward = bool(self.held & FORWARD_KEYS) - bool(self.held & BACK_KEYS)
        turn = bool(self.held & LEFT_KEYS) - bool(self.held & RIGHT_KEYS)
        v_max, w_max = C.MANUAL_FAST if self.held & SHIFT_KEYS else C.MANUAL_NORMAL
        if forward or turn:
            v = forward * v_max * (C.MANUAL_REVERSE_FACTOR if forward < 0 else 1.0)
            return Command(v, turn * w_max)
        if self.drag_origin is not None and self.drag_pos is not None:
            dx = (self.drag_pos[0] - self.drag_origin[0]) / DRAG_FULL_SCALE
            dy = (self.drag_pos[1] - self.drag_origin[1]) / DRAG_FULL_SCALE
            fwd = max(-1.0, min(1.0, -dy))
            v = fwd * v_max * (C.MANUAL_REVERSE_FACTOR if fwd < 0 else 1.0)
            return Command(v, max(-1.0, min(1.0, -dx)) * w_max)
        return Command()


class ManualDriver:
    """The keyboard/mouse driver behind the standard decide(observation) plug.
    Returns None when no drive input is held (no fresh command, so the robot stops)."""

    name = "human_manual"

    def __init__(self, manual_input: ManualInput):
        self.input = manual_input

    def decide(self, obs: Observation) -> Decision | None:
        if not self.input.drive_input_held:
            return None
        return Decision(self.input.command(), obs.seq)
