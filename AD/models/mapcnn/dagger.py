"""DAgger for the map CNN: the learned policy drives training layouts while the built-in driver,
fed the same observations, labels every state the policy reaches (its features and the map the
policy built there); the states are added to the recordings and the policy is fine-tuned.

    .venv\\Scripts\\python -m AD.models.mapcnn.dagger --base bc --run dagger --rounds 1 --lock ARE2-gpu

As AD/models/cnn/dagger.py: `episodes` training seeds per round (cats = seed % 5, as recorded),
the expert's command executed instead with probability `beta` (halved each round), then a
fine-tune, a benchmark and a log entry (run <run>_r<k>)."""

from __future__ import annotations

import argparse
import json
import random
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from ...common import dataset
from . import train

RUNS = train.RUNS


def rollout(args) -> dict:
    ckpt, seed, beta = args
    from robot_env import config as C
    from robot_env.baseline import BaselineDriver
    from robot_env.layout import RoomMap
    from robot_env.system import RobotSystem

    from ...common.features import LABEL_LEVEL, vector
    from .policy import MapDriver

    policy = MapDriver(ckpt)
    expert = BaselineDriver(C.SPEED_LEVELS[LABEL_LEVEL])
    system = RobotSystem(cats=seed % 5, cat_seed=seed)
    task = RoomMap(system.sim.model).sample_task(seed)
    system.reset(*task.start, task.goal)
    policy.reset()
    coin = random.Random(seed * 7919 + int(beta * 1000))
    X, M, V, outcome = [], [], [], "timeout"
    while system.time < C.EPISODE_TIME_LIMIT - 1e-9:
        obs = system.observe()
        if obs.goal_distance <= C.GOAL_RADIUS:
            outcome = "success"
            break
        label = expert.decide(obs)
        mine = policy.decide(obs)
        cmd = label.command if label else None
        X.append(vector(obs))
        M.append(policy.last_map)
        V.append((cmd.v, cmd.omega) if cmd else (0.0, 0.0))
        system.apply(label if (label is not None and coin.random() < beta) else mine)
        system.advance(C.DECISION_PERIOD)
        if system.collisions:
            outcome = "collision"
            break
    system.close()
    return {"seed": seed, "outcome": outcome, "x": np.array(X, np.float32), "maps": np.array(M, np.uint8),
            "cmd": np.array(V, np.float32)}


def collect(ckpt: str, seeds: list[int], beta: float, workers: int, out: Path) -> dict:
    with ProcessPoolExecutor(workers, max_tasks_per_child=1) as pool:  # a fresh process per episode (sim memory)
        eps = list(pool.map(rollout, [(ckpt, s, beta) for s in seeds]))
    np.savez_compressed(out, x=np.concatenate([e["x"] for e in eps]), maps=np.concatenate([e["maps"] for e in eps]),
                        cmd=np.concatenate([e["cmd"] for e in eps]))
    return {"states": int(sum(len(e["x"]) for e in eps)),
            "outcomes": {k: sum(e["outcome"] == k for e in eps) for k in ("success", "collision", "timeout")}}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--base", required=True, help="run name of the starting policy (runs/<base>/model.pt)")
    p.add_argument("--run", required=True)
    p.add_argument("--rounds", type=int, default=1)
    p.add_argument("--episodes", type=int, default=80)
    p.add_argument("--beta", type=float, default=0.5, help="expert share in round 1, halved each round")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--first-seed", type=int, default=0, help="offset into the training seeds")
    p.add_argument("--lock", default=None, help="hold the machine lock as this owner while retraining")
    a = p.parse_args(argv)
    train.check_memory()
    pool = list(dataset.TRAIN)
    files, ckpt, beta = [], str(RUNS / a.base / "model.pt"), a.beta
    for k in range(1, a.rounds + 1):
        seeds = [pool[(a.first_seed + (k - 1) * a.episodes + i) % len(pool)] for i in range(a.episodes)]
        out = RUNS / a.run / f"states_r{k}.npz"
        out.parent.mkdir(parents=True, exist_ok=True)
        meta = out.with_suffix(".json")
        if out.exists() and meta.exists():  # collected before (a rerun): retrain from it
            info = json.loads(meta.read_text(encoding="utf-8"))
        else:
            t0 = time.time()
            info = collect(ckpt, seeds, beta, a.workers, out)
            meta.write_text(json.dumps(info), encoding="utf-8")
            print(f"round {k}: collected {info} in {time.time() - t0:.0f} s (beta {beta:.2f})", flush=True)
        files.append(str(out))
        run = f"{a.run}_r{k}"
        train.main(["--run", run, "--init", ckpt, "--epochs", str(a.epochs), "--lr", "5e-4",
                    "--workers", str(a.workers), *(["--lock", a.lock] if a.lock else []), "--extra", *files,
                    "--notes", f"DAgger round {k} from {a.base}: +{info['states']} states (policy episodes "
                               f"{info['outcomes']}, beta {beta:.2f})"])
        ckpt, beta = str(RUNS / run / "model.pt"), beta / 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
