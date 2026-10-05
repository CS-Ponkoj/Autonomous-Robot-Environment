"""Check that the cats roam the whole floor safely (development gate, no rendering).

For each cat seed (robot parked at the seed-1000 start) it records, over the run:
  - when the herd together has been in all 5 regions (gate: within 180 s in every seed);
  - when each cat alone has been in all 5 regions (gate: within 600 s in at least 19 of 20 seeds);
  - the longest stall: a cat that wants to move (walk, travel, dart, flee, yield) but has not
    moved 5 cm (gate: never longer than 5 s);
  - the smallest gaps seen at any physics step (walls 0.03 m, robot 0.30 m, cats 0.05 m, within
    the agreed 1 mm) and contacts with the robot (gate: none);
  - how often a sample was rejected (the animated pose came too close) and how often a cat worked
    its way out of a tight spot (reported).
With --repeat, one seed is run twice and the two traces compared (gate: identical): a SHA-256
digest of every cat's interpolated colliders at every physics step and of its state, commands,
root pose, and speeds at every control tick.

    .venv\\Scripts\\python tools\\roam_check.py              (20 seeds, 3 cats, 600 s)
    .venv\\Scripts\\python tools\\roam_check.py --cats 4 --seeds 0-9 --seconds 300
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

HERD_LIMIT = 180.0  # s
CAT_LIMIT = 600.0  # s
STALL_LIMIT = 5.0  # s
GAP_TOLERANCE = 0.001  # m
STEP = 0.5  # s between coverage and stall samples
MOVING = ("walk", "travel", "dart", "flee", "yield")


def run_seed(args: tuple[int, int, float]) -> dict:
    seed, cats, seconds = args
    from robot_env import config as C
    from robot_env.cats import CAT_GAP, ROBOT_GAP, WALL_GAP
    from robot_env.layout import RoomMap
    from robot_env.system import RobotSystem

    s = RobotSystem(cats=cats, cat_seed=seed)
    task = RoomMap(s.sim.model).sample_task(1000)
    s.reset(*task.start, task.goal)
    herd = s.cats
    regions = len(C.ROOMS)
    herd_t, cat_t = None, [None] * cats
    still_from = [None] * cats
    longest = 0.0
    low = [math.inf, math.inf, math.inf]
    trace = hashlib.sha256()
    per_step = round(STEP / C.PHYSICS_DT)
    per_tick = round(C.CONTROL_PERIOD / C.PHYSICS_DT)
    while s.time < seconds - 1e-9:
        for k in range(per_step):  # every physics step: all cats at the same instant
            s.advance(C.PHYSICS_DT)
            for g in herd.current_gaps():
                low = [min(low[0], g["wall"]), min(low[1], g["robot"]), min(low[2], g["cat"])]
            trace.update(herd.col_world.tobytes())  # every collider as placed in the physics
            if k % per_tick == per_tick - 1:  # each control tick: what every cat decided
                for c in herd.cats:
                    trace.update(repr((c.state, c.cmd_v, c.cmd_w, c.cmd_lat, c.x, c.y, c.yaw, c.v, c.w,
                                       c.frozen, c.held)).encode())
        group = set()
        for i, c in enumerate(herd.cats):
            group |= c.visited
            if cat_t[i] is None and len(c.visited) >= regions:
                cat_t[i] = s.time
            if c.state not in MOVING or c.frozen:
                still_from[i] = None
                continue
            ref = still_from[i]
            if ref is None or math.hypot(c.x - ref[1], c.y - ref[2]) > 0.05:
                still_from[i] = (s.time, c.x, c.y)
            else:
                longest = max(longest, s.time - ref[0])
        if herd_t is None and len(group) >= regions:
            herd_t = s.time
    result = {"seed": seed, "herd": herd_t, "cats": cat_t, "stall": longest,
              "gaps": [low[0] - WALL_GAP, low[1] - ROBOT_GAP, low[2] - CAT_GAP],
              "contacts": s.cat_contacts, "held": sum(c.held for c in herd.cats),
              "escapes": sum(c.escaped for c in herd.cats), "trace": trace.hexdigest()}
    s.close()
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cats", type=int, default=3)
    ap.add_argument("--seeds", default="0-19", help="a range a-b or a comma list")
    ap.add_argument("--seconds", type=float, default=CAT_LIMIT)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--repeat", action="store_true", help="also run the first seed twice and compare")
    ap.add_argument("--strict", action="store_true", help="release gate: refuse unless the commit is known, the tree clean, and nothing changes during the run")
    ap.add_argument("--out", type=Path, help="write the result (with its provenance) as JSON here")
    a = ap.parse_args()
    from robot_env import provenance
    before = provenance.begin(__file__, sys.argv, a.strict, a.out,
                              model_sha256=provenance.model_of(cats=a.cats, cat_seed=0))
    if "-" in a.seeds:
        lo, hi = (int(v) for v in a.seeds.split("-"))
        seeds = list(range(lo, hi + 1))
    else:
        seeds = [int(v) for v in a.seeds.split(",")]
    jobs = [(sd, a.cats, a.seconds) for sd in seeds] + ([(seeds[0], a.cats, a.seconds)] if a.repeat else [])
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        results = list(pool.map(run_seed, jobs))
    repeat = results.pop() if a.repeat else None
    herd_ok = cat_ok = 0
    worst_stall, worst_gap, contacts = 0.0, math.inf, 0
    fmt = lambda t: "-" if t is None else f"{t:.0f}"  # noqa: E731
    for r in results:
        herd_ok += r["herd"] is not None and r["herd"] <= HERD_LIMIT
        cat_ok += all(t is not None and t <= CAT_LIMIT for t in r["cats"])
        worst_stall = max(worst_stall, r["stall"])
        worst_gap = min(worst_gap, min(r["gaps"]))
        contacts += r["contacts"]
        print(f"seed {r['seed']:3d}  herd {fmt(r['herd']):>4} s  cats {[fmt(t) for t in r['cats']]}  "
              f"stall {r['stall']:.1f} s  gap margins mm {[round(1000 * g, 2) for g in r['gaps']]}  "
              f"contacts {r['contacts']}  rejected samples {r['held']}  escapes {r['escapes']}")
    n = len(results)
    need_cat = math.ceil(0.95 * n)
    full = a.seconds >= CAT_LIMIT
    same = repeat is None or repeat["trace"] == results[0]["trace"]
    passed = (herd_ok == n and (not full or cat_ok >= need_cat) and worst_stall <= STALL_LIMIT
              and worst_gap >= -GAP_TOLERANCE and contacts == 0 and same)
    print(f"herd all regions <= {HERD_LIMIT:.0f} s: {herd_ok}/{n} (need {n}); "
          f"each cat <= {CAT_LIMIT:.0f} s: {cat_ok}/{n} (need {need_cat}{'' if full else ', not judged: short run'}); "
          f"longest stall {worst_stall:.1f} s (max {STALL_LIMIT:.0f}); smallest gap margin "
          f"{1000 * worst_gap:.2f} mm (min -1); contacts {contacts}; "
          f"repeat identical: {'-' if repeat is None else same}: {'PASS' if passed else 'FAIL'}")
    if repeat is not None:
        print(f"trace digest seed {results[0]['seed']}: {results[0]['trace']} / repeat {repeat['trace']}")
    after = provenance.end(before, __file__, sys.argv, a.strict, a.out,
                           model_sha256=provenance.model_of(cats=a.cats, cat_seed=0))
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps({**after, "result": "PASS" if passed else "FAIL", "seeds": [r["seed"] for r in results],
                                     "cats": a.cats, "seconds": a.seconds, "herd_ok": herd_ok, "cat_ok": cat_ok,
                                     "longest_stall_s": worst_stall, "smallest_gap_margin_mm": round(1000 * worst_gap, 2),
                                     "contacts": contacts, "repeat_identical": None if repeat is None else same,
                                     "per_seed": [{k: v for k, v in r.items()} for r in results]},
                                    indent=1, default=str), encoding="utf-8")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
