"""The held-out test (goal seeds 1000-1009, without cats and with 4 cats), run once for a
finished model and attached to its training entry.

    .venv\\Scripts\\python -m AD.models.cnn.final --model cnn --run dagger_disc_r3
    .venv\\Scripts\\python -m AD.models.cnn.final --model dt --run dt_k20"""

from __future__ import annotations

import argparse
import functools

from ...common import benchmark, tracker


def driver_factory(model: str, run: str, device: str | None = None):
    """A picklable driver maker for a saved run; device "cpu" keeps the GPU context out of
    every benchmark worker (None: the GPU if there is one)."""
    path = str(tracker.AD / "models" / model / "runs" / run / "model.pt")
    if model in ("cnn", "ppo"):  # a PPO checkpoint is saved as a continuous CNN
        from .policy import CNNDriver
        return functools.partial(CNNDriver, path, device=device)
    if model == "dt":
        from ..dt.train import DTDriver
        return functools.partial(DTDriver, path, device=device)
    raise ValueError(f"unknown model {model}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", required=True, choices=("cnn", "dt", "ppo"))
    p.add_argument("--run", required=True)
    p.add_argument("--workers", type=int, default=4)
    a = p.parse_args(argv)
    if any(e["model"] == a.model and e["run"] == a.run and e.get("final") for e in tracker.entries()):
        p.error("this run already has its held-out result (the test is run once)")
    final = benchmark.run(driver_factory(a.model, a.run), workers=a.workers, final=True)
    score = benchmark.score(final)
    tracker.add_final(a.model, a.run, final, score)
    print({k: final[k] for k in ("no_cats", "4_cats", "latency_ms")}, "final score", score, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
