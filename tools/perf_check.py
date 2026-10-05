"""Headless simulation cost with four cats (development gate, no rendering).

The robot drives among four cats (cat seed 16, turning one way then the other) for 30 simulated
seconds after a 2 s warm-up, three times. Reports the real-time factor (simulated seconds per
wall-clock second) of every run and each cat plan's time (count, p50, p95, worst) in a timed run.
Gate: the kernels compiled, the real-time factor at least 0.95 in every run, and plan p95 at most
1 ms. Writes qa_output/perf_check.json with the source revision (commit, clean or modified, and
the SHA-256 of every source file), the command, the host, and the result.

    .venv\\Scripts\\python tools\\perf_check.py
"""

from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

RTF_MIN = 0.95
PLAN_P95_MAX = 0.001  # s
SECONDS = 30.0


def run(timed: bool) -> tuple[float, list[float]]:
    import robot_env.cats as K
    from robot_env.system import RobotSystem
    plans: list[float] = []
    original = K.CatHerd._plan
    if timed:
        def plan(self, cat):
            t = time.perf_counter()
            original(self, cat)
            plans.append(time.perf_counter() - t)
        K.CatHerd._plan = plan
    try:
        s = RobotSystem(cats=4, cat_seed=16)
        s.reset(-4.0, -2.5, 0.0, (4.0, 2.5))
        s.advance(2.0)
        plans.clear()
        t = time.perf_counter()
        for k in range(int(SECONDS / 0.02)):
            if k % 5 == 0:
                s.drive(0.5, 0.3 if (k // 100) % 2 else -0.3)
            s.advance(0.02)
        wall = time.perf_counter() - t
        s.close()
    finally:
        K.CatHerd._plan = original
    return SECONDS / wall, plans


def main() -> int:
    import argparse

    from fps_protocol import host_info

    from robot_env import kernels, provenance
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--strict", action="store_true", help="release gate: refuse unless the commit is known, the tree clean, and nothing changes during the run")
    a = ap.parse_args()
    out = Path(__file__).resolve().parent.parent / "qa_output" / "perf_check.json"
    model = provenance.model_of(cats=4, cat_seed=16)
    before = provenance.begin(__file__, sys.argv, a.strict, out, model_sha256=model)
    warm = kernels.warm()
    rtfs = [run(False)[0] for _ in range(3)]
    _, plans = run(True)
    p = 1000 * np.array(plans)
    p95 = float(np.percentile(p, 95))
    passed = kernels.COMPILED and min(rtfs) >= RTF_MIN and p95 <= 1000 * PLAN_P95_MAX
    after = provenance.end(before, __file__, sys.argv, a.strict, out, model_sha256=provenance.model_of(cats=4, cat_seed=16))
    result = {**after, "command": "python tools/perf_check.py " + " ".join(sys.argv[1:]),
              "time": time.strftime("%Y-%m-%d %H:%M:%S"), "host": host_info(), "python": platform.python_version(),
              "kernels": {"compiled": kernels.COMPILED, "warm_s": round(warm, 2)},
              "scenario": {"cats": 4, "cat_seed": 16, "seconds": SECONDS, "warmup_s": 2.0, "runs": 3},
              "rtf": [round(r, 3) for r in rtfs],
              "plans": {"count": len(p), "p50_ms": round(float(np.percentile(p, 50)), 3), "p95_ms": round(p95, 3),
                        "worst_ms": round(float(p.max()), 3)},
              "gate": {"compiled": True, "rtf_min": RTF_MIN, "plan_p95_ms_max": 1000 * PLAN_P95_MAX,
                       "result": "PASS" if passed else "FAIL"}}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    rev = result["build"]
    print(f"4 cats, robot driving: real-time factor {', '.join(f'{r:.2f}' for r in rtfs)} (every run >= {RTF_MIN}); "
          f"{len(p)} cat plans: p50 {np.percentile(p, 50):.3f} p95 {p95:.3f} worst {p.max():.3f} ms "
          f"(p95 <= {1000 * PLAN_P95_MAX:.0f} ms); kernels compiled: {kernels.COMPILED}; revision {rev['commit']} "
          f"({rev['working_tree']}): {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
