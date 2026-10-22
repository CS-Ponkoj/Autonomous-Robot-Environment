"""Benchmark the built-in driver with a saved map (the new reference for map-planner models).

    python -m AD.models.mapplan.bench --map map_v1 --seeds 2000-2009
    python -m AD.models.mapplan.bench --map floorplan --seeds 2000-2009     (perfect-map comparison)
    python -m AD.models.mapplan.bench --map map_v1 --final                  (held-out, once)

Logged as model "builtin", run "map_<name>"; --final adds the held-out result to that run."""

from __future__ import annotations

import argparse
import functools

from ...common import benchmark, tracker
from .driver import MapDriver
from .savedmap import SavedMap

RUNS = tracker.AD / "models" / "mapplan" / "runs"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--map", default="map_v1", help="a map folder under AD/models/mapplan/runs")
    p.add_argument("--seeds", default="2000-2009", help="practice seed range a-b")
    p.add_argument("--final", action="store_true", help="the held-out test seeds (run once per finished model)")
    p.add_argument("--match", action="store_true", help="correct odometry drift by matching scans to the map")
    p.add_argument("--odo-bias", type=float, nargs=2, default=(0.0, 0.0), metavar=("SPEED", "YAWRATE"),
                   help="test only: odometry error added inside the driver (e.g. 0.02 0.0087 = 2%% speed, 0.5 deg/s)")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--no-log", action="store_true")
    a = p.parse_args(argv)
    path = RUNS / a.map / "map.npz"
    SavedMap.load(path)  # fail here, not in every worker, if the map is missing
    make = functools.partial(MapDriver, str(path), match=a.match, odo_bias=tuple(a.odo_bias))
    run = f"map_{a.map}" + ("_match" if a.match else "") + ("_bias" if any(a.odo_bias) else "")
    if a.final:
        if not any(e["model"] == "builtin" and e["run"] == run for e in tracker.entries()):
            p.error(f"log the practice run {run} first (the held-out result is added to it)")
        if any(e["model"] == "builtin" and e["run"] == run and e.get("final") for e in tracker.entries()):
            p.error("this run already has its held-out result (the test is run once)")
        final = benchmark.run(make, workers=a.workers, final=True)
        score = benchmark.score(final)
        print({k: final[k] for k in ("no_cats", "4_cats", "latency_ms")}, "final score", score, flush=True)
        if not a.no_log:
            tracker.add_final("builtin", run, final, score)
        return 0
    lo, hi = (int(v) for v in a.seeds.split("-"))
    summary = benchmark.run(make, workers=a.workers, seeds=tuple(range(lo, hi + 1)))
    entry = {"model": "builtin", "run": run, "kind": "benchmark", "base": "-", "data": f"saved map {a.map}",
             "params": {"map": a.map, "match": a.match, "odo_bias": list(a.odo_bias), "seeds": a.seeds}, "bench": summary, "score": benchmark.score(summary),
             "notes": "built-in driver starting from the saved building map"
                      + (" (simulator floor plan: comparison only)" if a.map == "floorplan" else "")}
    print({k: summary[k] for k in ("no_cats", "4_cats", "latency_ms")}, "score", entry["score"], flush=True)
    if not a.no_log:
        tracker.log(entry)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
