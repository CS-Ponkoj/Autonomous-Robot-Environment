"""Score a PPO candidate checkpoint on the practice drives WITHOUT logging it as a model (choosing
which checkpoints to keep); the result goes to runs/<run>/practice.json.

    .venv\Scripts\python -m AD.models.ppo.evalcand --model hybrid --run hppo_c1 --workers 1"""

from __future__ import annotations

import argparse
import json

from ...common import benchmark, tracker
from ..cnn.final import driver_factory


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", default="hybrid", choices=("ppo", "hybrid"))
    p.add_argument("--run", required=True)
    p.add_argument("--workers", type=int, default=1)
    a = p.parse_args(argv)
    s = benchmark.run(driver_factory(a.model, a.run, "cpu"), workers=a.workers)
    mins = sum(e["time_s"] for e in s["per_episode"]) / 60
    out = {"score": benchmark.score(s), "goals": s["success_total"], "no_cats": s["no_cats"]["success"],
           "cats": s["4_cats"]["success"], "collisions": s["collisions_total"],
           "interventions_per_min": round((s["no_cats"]["interventions"] + s["4_cats"]["interventions"]) / mins, 2),
           "latency_ms": s["latency_ms"]}
    (tracker.AD / "models" / a.model / "runs" / a.run / "practice.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(a.run, out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
