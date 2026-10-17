"""Benchmark a PPO checkpoint (the 20 practice drives, CPU drivers) and log it as its own run.

    .venv\Scripts\python -m AD.models.ppo.bench --run ppo_r1"""

from __future__ import annotations

import argparse
import functools
import json

from ...common import benchmark, tracker
from ..cnn.policy import CNNDriver
from .train import RUNS


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--run", required=True)
    p.add_argument("--workers", type=int, default=2)
    a = p.parse_args(argv)
    meta = json.loads((RUNS / a.run / "meta.json").read_text(encoding="utf-8"))
    summary = benchmark.run(functools.partial(CNNDriver, str(RUNS / a.run / "model.pt"), device="cpu"), workers=a.workers)
    entry = tracker.log({"model": "ppo", "run": a.run, "kind": "finetune", "base": meta["base"],
                         "data": f"{meta['timesteps']} PPO steps in robot_env ({meta['envs']} envs, cats {meta['cats']})",
                         "params": meta["params"], "train_s": meta["train_s"], "bench": summary,
                         "score": benchmark.score(summary),
                         "notes": f"PPO from {meta['base']}; training return {meta['recent_return']}, "
                                  f"training success {meta['recent_success']} (last {meta['recent_episodes']} episodes)"})
    print({k: summary[k] for k in ("no_cats", "4_cats", "latency_ms")}, "score", entry["score"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
