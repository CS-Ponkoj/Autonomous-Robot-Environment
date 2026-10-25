"""Benchmark existing weights again on a wider practice set (never the held-out test seeds), to
compare models on more drives than the standard 20. No training: an evaluation, logged as its own
benchmark entry (run "<run>@<first>-<last>").

    .venv\Scripts\python -m AD.models.cnn.rebench --model cnn --run dagger_cont_r3 --seeds 2000-2019
    .venv\Scripts\python -m AD.models.cnn.rebench --model dt --run dt_noact --seeds 2000-2019 --workers 2"""

from __future__ import annotations

import argparse

from ...common import benchmark, tracker
from .final import MODELS, driver_factory


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", required=True, choices=MODELS)
    p.add_argument("--run", required=True)
    p.add_argument("--seeds", default="2000-2019", help="practice seed range a-b")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", default="cpu", choices=("cpu", "cuda"), help="where each worker runs the model")
    p.add_argument("--tag", default=None, help="name the entry <run>@<tag> instead of <run>@<first>-<last>")
    p.add_argument("--route", default=None, help="route-input runs: the route source (default: the one trained with)")
    a = p.parse_args(argv)
    lo, hi = (int(v) for v in a.seeds.split("-"))
    seeds = tuple(range(lo, hi + 1))
    if set(seeds) & set(benchmark.HELDOUT):
        p.error("these seeds include held-out test seeds; use final.py for the test (run once)")
    summary = benchmark.run(driver_factory(a.model, a.run, a.device, a.route), workers=a.workers, seeds=seeds)
    score = benchmark.score(summary)
    tracker.log({"model": a.model, "run": f"{a.run}@{a.tag or f'{lo}-{hi}'}", "kind": "benchmark", "base": a.run,
                 "bench": summary, "score": score,
                 "notes": f"re-benchmark of {a.run} on practice seeds {lo}-{hi} ({len(seeds) * 2} drives)"
                          + (f" [{a.tag}]" if a.tag else "")})
    print({k: summary[k] for k in ("no_cats", "4_cats", "latency_ms")}, "score", score, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
