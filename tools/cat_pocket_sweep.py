"""Cat sweep around the office desk and chair (development check; tests/test_furniture.py has
spot checks):

    .venv\\Scripts\\python tools\\cat_pocket_sweep.py [probes]

Every tight start pose (wall gap within 5 cm above WALL_GAP) in the box x -4.7..-2.7, y 3.0..4.95,
8 headings, one per 0.1 m cell and heading, then an even sample of `probes` of them (default 60).
Each: a cat placed walking there, 15 s; the longest time it wants to move (walk, travel, dart,
flee, yield) without moving 5 cm, sampled every 0.5 s like tools/roam_check.py, must stay within
5 s, and every gap must be kept. Prints the failures and the count."""
import math
import sys

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
from robot_env.cats import CAT_GAP, ROBOT_GAP, WALL_GAP  # noqa: E402
from robot_env.system import RobotSystem  # noqa: E402

s = RobotSystem(cats=1, cat_seed=2)
s.reset(4.0, -4.0, 0.0, (3.0, -3.0))
herd = s.cats
cands, seen = [], set()
for x in np.arange(-4.7, -2.69, 0.05):
    for y in np.arange(3.0, 4.96, 0.05):
        for h in range(0, 360, 45):
            herd.place(0, float(x), float(y), math.radians(h), state="pause")
            g = herd.current_gaps()[0]["wall"]
            key = (round(x / 0.1), round(y / 0.1), h)
            if WALL_GAP <= g <= WALL_GAP + 0.05 and key not in seen:
                seen.add(key)
                cands.append((round(float(x), 3), round(float(y), 3), float(h)))
s.close()
limit = int(sys.argv[1]) if len(sys.argv) > 1 else 60
step = max(1, len(cands) // limit)
picked = cands[::step][:limit]
print("tight poses", len(cands), "probing", len(picked))
sys.stdout.flush()
fails = 0
for x, y, h in picked:
    s = RobotSystem(cats=1, cat_seed=2)
    s.reset(4.0, -4.0, 0.0, (3.0, -3.0))
    herd = s.cats
    herd.place(0, x, y, math.radians(h), state="walk")
    cat = herd.cats[0]
    start = (cat.x, cat.y)
    low = [math.inf] * 3
    ref, stall = None, 0.0
    for k in range(30):  # 15 s, sampled like tools/roam_check.py (0.5 s)
        s.advance(0.5)
        g = herd.current_gaps()[0]
        low = [min(low[0], g["wall"]), min(low[1], g["robot"]), min(low[2], g["cat"])]
        moving = cat.state in ("walk", "travel", "dart", "flee", "yield")
        if not moving:
            ref = None
        elif ref is None or math.hypot(cat.x - ref[1], cat.y - ref[2]) > 0.05:
            ref = (herd.time, cat.x, cat.y)
        else:
            stall = max(stall, herd.time - ref[0])
    moved = math.hypot(cat.x - start[0], cat.y - start[1])
    s.close()
    ok = stall <= 5.0 and low[0] >= WALL_GAP - 0.001 and low[1] >= ROBOT_GAP - 0.001 and low[2] >= CAT_GAP - 0.001
    if not ok:
        fails += 1
        print(f"FAIL ({x}, {y}, {h}) stall {stall:.1f} s moved {moved:.3f} low {[round(v, 4) for v in low]}")
        sys.stdout.flush()
print("failures", fails, "of", len(picked))
