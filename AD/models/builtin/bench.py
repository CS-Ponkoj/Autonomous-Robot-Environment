"""Benchmark the built-in rule-based driver at the label speed level (the reference row).
It is not tuned here, so it runs both the practice and the held-out test layouts.

    .venv\Scripts\python -m AD.models.builtin.bench --workers 10"""

from __future__ import annotations

from robot_env import config as C
from robot_env.baseline import BaselineDriver

from ...common import benchmark, tracker
from ...common.features import LABEL_LEVEL


def make_driver():
    return BaselineDriver(C.SPEED_LEVELS[LABEL_LEVEL])


def main(argv=None) -> int:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=10)
    p.add_argument("--seeds", help="practice re-benchmark on seeds a-b only (logged as reference@a-b; never the test seeds)")
    a = p.parse_args(argv)
    if a.seeds:
        lo, hi = (int(v) for v in a.seeds.split("-"))
        seeds = tuple(range(lo, hi + 1))
        if set(seeds) & set(benchmark.HELDOUT):
            p.error("the held-out test seeds are not for re-benchmarks")
        s = benchmark.run(make_driver, workers=a.workers, seeds=seeds)
        tracker.log({"model": "builtin", "run": f"reference@{a.seeds}", "kind": "benchmark", "base": "reference",
                     "bench": s, "score": benchmark.score(s), "notes": f"practice re-benchmark (seeds {a.seeds})"})
        print("builtin", benchmark.score(s), s["success_total"], "/", s["of_total"])
        return 0
    dev = benchmark.run(make_driver, workers=a.workers)
    final = benchmark.run(make_driver, workers=a.workers, final=True)
    entry = tracker.log({"model": "builtin", "run": "reference", "kind": "benchmark", "bench": dev,
                         "score": benchmark.score(dev), "final": final, "final_score": benchmark.score(final),
                         "offline": {"accuracy": 1.0}, "notes": "the reference: its own actions are the labels"})
    print("practice", entry["score"], "test", entry["final_score"], final["no_cats"], final["4_cats"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
