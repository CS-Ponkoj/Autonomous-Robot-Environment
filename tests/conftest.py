"""Validation hook for a future speed level that is not exposed yet.

With ROBOT_CANDIDATE_LEVELS="1.2:1.5" (m/s:rad/s, comma separated) the whole suite runs as if
those levels were appended to SPEED_LEVELS (and the speed cap follows). Candidates must be
faster than every exposed level (the levels stay one ordered, contiguous list). The production
configuration in robot_env/config.py is not changed; a level is listed there only after it
passes this run. Levels 4 and 5 (0.7 and 1.0 m/s) were validated this way and are exposed.
"""

import os

from robot_env import config as C

_CANDIDATES = os.environ.get("ROBOT_CANDIDATE_LEVELS", "").strip()
if _CANDIDATES:
    extra = tuple(tuple(float(x) for x in item.split(":")) for item in _CANDIDATES.split(","))
    combined = C.SPEED_LEVELS + extra
    speeds = [v for v, _ in combined]
    if any(b <= a for a, b in zip(speeds, speeds[1:])):
        raise ValueError(f"candidate levels must be faster than every exposed level: {combined}")
    C.SPEED_LEVELS = combined
    C.MAX_LINEAR_SPEED = max(speeds)


import functools  # noqa: E402

import pytest  # noqa: E402


@functools.cache
def _no_display() -> str | None:
    """Why no desktop window can open here, or None. Only this check can skip a gui test."""
    import pygame
    try:
        pygame.display.init()
        n = pygame.display.get_num_displays()
    except pygame.error as e:
        return f"no display: {e}"
    finally:
        pygame.display.quit()
    return None if n > 0 else "no display attached"


@functools.cache
def _no_opengl() -> str | None:
    """Why no OpenGL context can be created here, or None. Only this check can skip an opengl test."""
    import mujoco
    try:
        ctx = mujoco.GLContext(16, 16)
        ctx.make_current()
        ctx.free()
    except Exception as e:  # context creation only: rendering failures still fail their tests
        return f"no OpenGL context: {type(e).__name__}: {e}"
    return None


def pytest_collection_modifyitems(config, items):
    for item in items:
        if item.get_closest_marker("gui") and (why := _no_display() or _no_opengl()):
            item.add_marker(pytest.mark.skip(reason=why))
        elif item.get_closest_marker("opengl") and (why := _no_opengl()):
            item.add_marker(pytest.mark.skip(reason=why))
