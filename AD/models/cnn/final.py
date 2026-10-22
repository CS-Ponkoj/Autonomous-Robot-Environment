"""The held-out test (goal seeds 1000-1009, without cats and with 4 cats), run once for a
finished model and attached to its training entry.

    .venv\\Scripts\\python -m AD.models.cnn.final --model cnn --run dagger_disc_r3
    .venv\\Scripts\\python -m AD.models.cnn.final --model dt --run dt_k20"""

from __future__ import annotations

import argparse
import functools

from ...common import benchmark, tracker


def driver_factory(model: str, run: str, device: str | None = None, route: str | None = None):
    """A picklable driver maker for a saved run; device "cpu" keeps the GPU context out of
    every benchmark worker (None: the GPU if there is one). A route-input run (every hybrid run,
    or any checkpoint with a route spec) gets its family's route driver; route: run it with
    another route source than the one it was trained with (e.g. "map")."""
    path = str(tracker.AD / "models" / model / "runs" / run / "model.pt")
    import torch
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if model == "hybrid" or "route_spec" in ck:
        import importlib
        routed = importlib.import_module(f"AD.models.{model}.routed")
        return functools.partial(routed.Driver, path, device or "cpu", route)
    if model == "mlp":
        from ..mlp.model import MLPDriver
        return functools.partial(MLPDriver, path, device=device)
    if model == "mapcnn":
        from ..mapcnn.policy import MapDriver
        return functools.partial(MapDriver, path, device=device or "cpu")
    if model in ("cnn", "ppo"):  # a PPO checkpoint is saved as a continuous CNN
        from .policy import CNNDriver
        return functools.partial(CNNDriver, path, device=device)
    if model == "dt":
        from ..dt.train import DTDriver
        return functools.partial(DTDriver, path, device=device)
    if model == "cfc":
        from ..cfc.model import CfCDriver
        return functools.partial(CfCDriver, path, device=device)
    raise ValueError(f"unknown model {model}")


MODELS = ("cnn", "dt", "ppo", "cfc", "hybrid", "mlp", "mapcnn")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", required=True, choices=MODELS)
    p.add_argument("--run", required=True)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", default="cpu", choices=("cpu", "cuda"), help="where each worker runs the model")
    p.add_argument("--route", default=None, help="route-input runs: the route source (default: the one trained with)")
    a = p.parse_args(argv)
    if any(e["model"] == a.model and e["run"] == a.run and e.get("final") for e in tracker.entries()):
        p.error("this run already has its held-out result (the test is run once)")
    final = benchmark.run(driver_factory(a.model, a.run, a.device, a.route), workers=a.workers, final=True)
    score = benchmark.score(final)
    tracker.add_final(a.model, a.run, final, score)
    print({k: final[k] for k in ("no_cats", "4_cats", "latency_ms")}, "final score", score, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
