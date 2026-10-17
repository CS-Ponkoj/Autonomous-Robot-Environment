"""DAgger for the CNN policy: the learned policy drives training layouts while the built-in
driver, fed the same observations, labels every state the policy actually reaches; the labels
are added to the data and the policy is retrained, round after round.

    .venv\\Scripts\\python -m AD.models.cnn.dagger --base bc_disc --rounds 4 --episodes 80 --run dagger_disc

Each round: `episodes` training seeds (cycling through 10000-10499, never a validation or test
seed; cats = seed % 5 as in the recording), the policy in control (with probability `beta` the
expert's command is executed instead, beta halving each round), then a fine-tune on the original
recordings plus every round's states, a benchmark, and a log entry (run <run>_r<k>)."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from ...common import dataset, tracker
from . import train
from .policy import CNNDriver, stack

RUNS = train.RUNS


def rollout(args) -> dict:
    """One episode with the policy driving and the built-in driver labelling every state.
    args: (checkpoint, seed, beta[, kind]); kind "cnn" (default) or "cfc" (a recurrent policy)."""
    ckpt, seed, beta = args[:3]
    kind = args[3] if len(args) > 3 else "cnn"
    from robot_env import config as C
    from robot_env.baseline import BaselineDriver
    from robot_env.layout import RoomMap
    from robot_env.system import RobotSystem

    from ...common.features import ACTION_INDEX, LABEL_LEVEL, action_of, vector

    if kind == "cfc":
        from ..cfc.model import CfCDriver
        policy = CfCDriver(ckpt, device="cpu")
    else:
        policy = CNNDriver(ckpt, device="cpu")  # many rollout workers: CPU (training stays on the GPU)
    expert = BaselineDriver(C.SPEED_LEVELS[LABEL_LEVEL])
    system = RobotSystem(cats=seed % 5, cat_seed=seed)
    task = RoomMap(system.sim.model).sample_task(seed)
    system.reset(*task.start, task.goal)
    policy.reset()
    coin = random.Random(seed * 7919 + int(beta * 1000))
    X, V, A, outcome = [], [], [], "timeout"
    while system.time < C.EPISODE_TIME_LIMIT - 1e-9:
        obs = system.observe()
        if obs.goal_distance <= C.GOAL_RADIUS:
            outcome = "success"
            break
        label = expert.decide(obs)
        mine = policy.decide(obs)
        cmd = label.command if label else None
        X.append(vector(obs))
        V.append((cmd.v, cmd.omega) if cmd else (0.0, 0.0))
        A.append(ACTION_INDEX[action_of(cmd)])
        system.apply(label if (label is not None and coin.random() < beta) else mine)
        system.advance(C.DECISION_PERIOD)
        if system.collisions:
            outcome = "collision"
            break
    system.close()
    return {"seed": seed, "outcome": outcome, "x": np.array(X, np.float32), "cmd": np.array(V, np.float32),
            "action": np.array(A, np.int8)}


def collect(ckpt: str, seeds: list[int], beta: float, frames: int, workers: int, out) -> dict:
    with ProcessPoolExecutor(workers, max_tasks_per_child=1) as pool:  # a fresh process per episode (sim memory)
        eps = list(pool.map(rollout, [(ckpt, s, beta) for s in seeds]))
    x = np.concatenate([stack(e["x"], np.full(len(e["x"]), e["seed"]), frames) for e in eps])
    np.savez_compressed(out, x=x, cmd=np.concatenate([e["cmd"] for e in eps]),
                        action=np.concatenate([e["action"] for e in eps]))
    return {"states": int(len(x)), "outcomes": {k: sum(e["outcome"] == k for e in eps)
                                                 for k in ("success", "collision", "timeout")}}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--base", required=True, help="run name of the starting policy (AD/cnn/runs/<base>/model.pt)")
    p.add_argument("--run", required=True)
    p.add_argument("--rounds", type=int, default=4)
    p.add_argument("--episodes", type=int, default=80)
    p.add_argument("--beta", type=float, default=0.5, help="expert share in round 1, halved each round")
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--first-seed", type=int, default=0, help="offset into the training seeds")
    p.add_argument("--lock", default=None, help="hold the machine lock as this owner while retraining")
    p.add_argument("--start-round", type=int, default=1, help="resume: earlier rounds' states and checkpoint are reused")
    a = p.parse_args(argv)
    from .policy import load
    base = RUNS / a.base / "model.pt"
    net = load(base)
    pool = list(dataset.TRAIN)
    files, ckpt, beta = [], str(base), a.beta
    for k in range(1, a.start_round):  # resume after these rounds
        files.append(str(RUNS / a.run / f"states_r{k}.npz"))
        ckpt, beta = str(RUNS / f"{a.run}_r{k}" / "model.pt"), beta / 2
        assert Path(files[-1]).exists() and Path(ckpt).exists(), f"round {k} is not complete"
    for k in range(a.start_round, a.rounds + 1):
        seeds = [pool[(a.first_seed + (k - 1) * a.episodes + i) % len(pool)] for i in range(a.episodes)]
        out = RUNS / a.run / f"states_r{k}.npz"
        out.parent.mkdir(parents=True, exist_ok=True)
        meta = out.with_suffix(".json")
        if out.exists() and meta.exists():  # collected before (a resumed run): retrain from it
            info = json.loads(meta.read_text(encoding="utf-8"))
            print(f"round {k}: reusing {info}", flush=True)
        else:
            t0 = time.time()
            info = collect(ckpt, seeds, beta, net.frames, a.workers, out)
            meta.write_text(json.dumps(info), encoding="utf-8")
            print(f"round {k}: collected {info} in {time.time() - t0:.0f} s (beta {beta:.2f})", flush=True)
        files.append(str(out))
        run = f"{a.run}_r{k}"
        train.main(["--head", net.head, "--frames", str(net.frames), "--width", str(net.width), "--run", run,
                    *([] if net.odometry else ["--no-odometry"]),
                    *(["--lock", a.lock] if a.lock else []), "--bench-device", "cpu", "--epochs", str(a.epochs), "--lr", "5e-4", "--init", ckpt, "--workers", str(a.workers), "--extra", *files,
                    "--notes", f"DAgger round {k} from {a.base}: +{info['states']} states (policy episodes "
                               f"{info['outcomes']}, beta {beta:.2f})"])
        ckpt, beta = str(RUNS / run / "model.pt"), beta / 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
