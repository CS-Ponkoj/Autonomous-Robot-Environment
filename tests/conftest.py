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
