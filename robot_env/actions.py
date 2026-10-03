"""The 5 discrete actions (starting test values) and the adapter to [v, omega]."""

from __future__ import annotations

from .types import Command

ACTIONS: dict[str, Command] = {
    "forward": Command(0.30, 0.0),
    "turn_left": Command(0.0, 1.0),
    "turn_right": Command(0.0, -1.0),
    "stop": Command(0.0, 0.0),
    "back_up": Command(-0.15, 0.0),
}
ACTION_NAMES: tuple[str, ...] = tuple(ACTIONS)


def to_command(action: str | int) -> Command:
    """Map an action name or index to its command."""
    if isinstance(action, int):
        return ACTIONS[ACTION_NAMES[action]]
    return ACTIONS[action]
