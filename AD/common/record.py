"""Record the built-in (reference) driver's decisions as training examples.

    .venv\\Scripts\\python -m AD.common.record --seeds 10000-10599 --workers 10

Each episode runs headless on a training goal seed (never a test seed), with 0 to 4 cats, and
saves every decision: the feature vector, the scene text, the driver's continuous command, and
its discrete action label. One file per seed in AD/data/ (git-ignored)."""

from __future__ import annotations

import argparse
import json
import math
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parent.parent / "data"
TEST_SEEDS = set(range(1000, 1010)) | set(range(2000, 2020)) | set(range(5000, 5030)) | {4002, 4016}


def parse_seeds(text: str) -> list[int]:
    if "-" in text:
        a, b = (int(v) for v in text.split("-"))
        return list(range(a, b + 1))
    return [int(v) for v in text.split(",")]


def record_seed(seed: int) -> dict:
    from robot_env import config as C
    from robot_env.baseline import BaselineDriver
    from robot_env.layout import RoomMap
    from robot_env.system import RobotSystem

    from .features import ACTION_INDEX, FEATURE_VERSION, LABEL_LEVEL, LABEL_VERSION, action_of, scene_text, vector

    cats = seed % 5  # 0 to 4 cats, spread evenly over the seeds
    system = RobotSystem(cats=cats, cat_seed=seed)
    task = RoomMap(system.sim.model).sample_task(seed)
    driver = BaselineDriver(C.SPEED_LEVELS[LABEL_LEVEL])
    system.reset(*task.start, task.goal)
    X, V, A, T, texts = [], [], [], [], []
    outcome = "timeout"
    while system.time < C.EPISODE_TIME_LIMIT - 1e-9:
        obs = system.observe()
        if obs.goal_distance <= C.GOAL_RADIUS:
            outcome = "success"
            break
        decision = driver.decide(obs)
        cmd = decision.command if decision else None
        X.append(vector(obs))
        V.append((cmd.v, cmd.omega) if cmd else (0.0, 0.0))
        A.append(ACTION_INDEX[action_of(cmd)])
        T.append(obs.time)
        texts.append(scene_text(obs))
        system.apply(decision)
        system.advance(C.DECISION_PERIOD)
        if system.collisions:
            outcome = "collision"
            break
    if outcome == "timeout" and system.observe().goal_distance <= C.GOAL_RADIUS:
        outcome = "success"  # reached on the very last tick
    system.close()
    DATA.mkdir(exist_ok=True)
    # The text first, the arrays last and through a temporary name: a seed counts as recorded
    # only once its .npz exists, so an interrupted write is simply recorded again.
    (DATA / f"seed{seed}.txt").write_text("\n".join(texts), encoding="utf-8")
    tmp = DATA / f"seed{seed}.tmp.npz"
    np.savez_compressed(tmp, x=np.array(X, np.float32), cmd=np.array(V, np.float32),
                        action=np.array(A, np.int8), t=np.array(T, np.float32),
                        feature_version=FEATURE_VERSION, label_version=LABEL_VERSION, outcome=outcome)
    tmp.replace(DATA / f"seed{seed}.npz")
    return {"seed": seed, "cats": cats, "outcome": outcome, "decisions": len(A), "time_s": round(system.time, 1)}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--seeds", default="10000-10599")
    p.add_argument("--workers", type=int, default=10)
    a = p.parse_args(argv)
    seeds = parse_seeds(a.seeds)
    if TEST_SEEDS & set(seeds):
        p.error("these seeds include test seeds; record only training seeds")
    todo = [s for s in seeds if not (DATA / f"seed{s}.npz").exists()]  # resumable
    start, results = time.time(), []
    # a fresh process per seed: a worker's memory grows over long episodes and is not given back
    with ProcessPoolExecutor(a.workers, max_tasks_per_child=1) as pool, \
            open(DATA / "episodes.jsonl", "a", encoding="utf-8") as log:
        for i, r in enumerate(pool.map(record_seed, todo), 1):
            results.append(r)
            log.write(json.dumps(r) + "\n")  # as each finishes, so an interrupted run keeps them
            log.flush()
            if i % 25 == 0 or i == len(todo):
                ok = sum(x["outcome"] == "success" for x in results)
                print(f"{i}/{len(todo)} seeds  {ok} reached  {time.time() - start:.0f} s", flush=True)
    n = sum(r["decisions"] for r in results)
    print(f"recorded {len(results)} episodes, {n} decisions ({math.floor(n / max(len(results), 1))} per episode)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
