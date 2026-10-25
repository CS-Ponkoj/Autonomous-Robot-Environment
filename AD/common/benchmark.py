"""The driving benchmark every model runs, through the environment's own driver plug and safety
layer, once without cats and once with 4 cats.

Two seed sets:
- DEV (2000-2009): for tuning and for every training run's score.
- HELDOUT (1000-1009): the final test, run once per finished model (run(..., final=True));
  scoring tuning runs on it would turn the test into tuning data.

Latency-aware: a decision is applied only after the time the model took to compute it, as on
a real robot. While the model thinks, the robot keeps executing its previous command (which
expires after robot_env's 0.3 s command lifetime and then stops), and a model slower than the
0.1 s decision period simply decides less often. A slow model therefore scores worse."""

from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

DEV = tuple(range(2000, 2010))
HELDOUT = tuple(range(1000, 1010))
CONDITIONS = ((0, "no cats"), (4, "4 cats"))


def _episode(args) -> dict:
    make_driver, seed, cats = args
    from robot_env import config as C
    from robot_env.layout import RoomMap
    from robot_env.system import RobotSystem

    driver = make_driver()
    system = RobotSystem(cats=cats, cat_seed=seed)
    task = RoomMap(system.sim.model).sample_task(seed)
    system.reset(*task.start, task.goal)
    reset = getattr(driver, "reset", None)
    if reset is not None:
        import inspect
        params = inspect.signature(reset).parameters
        if "initial_pose" in params or any(p.kind == p.VAR_KEYWORD for p in params.values()):
            reset(initial_pose=task.start)  # route and map drivers: the start pose on the map
        else:
            reset()
    start_goal, lat, outcome = system.observe().goal_distance, [], "timeout"
    tick = C.DECISION_PERIOD
    while system.time < C.EPISODE_TIME_LIMIT - 1e-9:
        obs = system.observe()
        if obs.goal_distance <= C.GOAL_RADIUS:
            outcome = "success"
            break
        t0 = time.perf_counter()
        decision = driver.decide(obs)
        took = time.perf_counter() - t0
        lat.append(took)
        # the world moves on while the model thinks: the previous command keeps running
        if took > 1e-3:
            system.advance(min(took, C.EPISODE_TIME_LIMIT - system.time))
        system.apply(decision)
        # then the rest of the decision period (a slow model decides again only after this)
        rest = tick - (took % tick) if took > 1e-3 else tick
        system.advance(min(rest, max(C.EPISODE_TIME_LIMIT - system.time, 0.0)))
        if system.collisions:
            outcome = "collision"
            break
    if outcome == "timeout" and system.observe().goal_distance <= C.GOAL_RADIUS:
        outcome = "success"
    r = {"seed": seed, "cats": cats, "outcome": outcome, "time_s": round(system.time, 2),
         "interventions": system.intervention_events, "progress_m": round(start_goal - system.observe().goal_distance, 2),
         "latency_ms": [round(1000 * v, 3) for v in lat]}
    system.close()
    return r


def run(make_driver, workers: int = 10, final: bool = False, seeds=None, conditions=CONDITIONS) -> dict:
    """Benchmark the driver made by make_driver() (a picklable, top-level callable: each episode
    gets a fresh driver) on DEV, or on HELDOUT with final=True. workers=1 runs one episode at a
    time (GPU models; it also keeps their measured latency free of CPU contention)."""
    seeds = seeds or (HELDOUT if final else DEV)
    jobs = [(make_driver, s, c) for c, _ in conditions for s in seeds]
    # Every episode in a fresh process, even with one worker: a simulation process does not give
    # its memory back between episodes (it grew to 9 GB over one serial benchmark). A GPU model
    # is loaded once per episode as a result (seconds), not per decision.
    with ProcessPoolExecutor(max(workers, 1), max_tasks_per_child=1) as pool:
        eps = list(pool.map(_episode, jobs))
    out = summarize(eps, conditions)
    out["set"] = "heldout" if seeds == HELDOUT else "dev" if seeds == DEV else "custom"
    return out


def summarize(eps: list[dict], conditions=CONDITIONS) -> dict:
    out = {"episodes": len(eps)}
    for cats, name in conditions:
        sel = [e for e in eps if e["cats"] == cats]
        key = name.replace(" ", "_")
        out[key] = {
            "success": sum(e["outcome"] == "success" for e in sel), "of": len(sel),
            "collisions": sum(e["outcome"] == "collision" for e in sel),
            "timeouts": sum(e["outcome"] == "timeout" for e in sel),
            "interventions": sum(e["interventions"] for e in sel),
            "mean_time_success_s": round(float(np.mean([e["time_s"] for e in sel if e["outcome"] == "success"])), 1)
            if any(e["outcome"] == "success" for e in sel) else None,
            "mean_progress_m": round(float(np.mean([e["progress_m"] for e in sel])), 2) if sel else None,
        }
    lat = np.concatenate([e["latency_ms"] for e in eps]) if eps else np.zeros(1)
    out["latency_ms"] = {"median": round(float(np.median(lat)), 2), "p95": round(float(np.percentile(lat, 95)), 2),
                         "late_share": round(float((lat > 100.0).mean()), 4)}
    out["success_total"] = sum(e["outcome"] == "success" for e in eps)
    out["collisions_total"] = sum(e["outcome"] == "collision" for e in eps)
    out["timeouts_total"] = sum(e["outcome"] == "timeout" for e in eps)
    out["interventions_total"] = sum(e["interventions"] for e in eps)
    out["of_total"] = len(eps)
    out["per_episode"] = [{k: v for k, v in e.items() if k != "latency_ms"} for e in eps]
    return out


def score(summary: dict) -> float:
    """One number to rank by: goals reached, minus collisions counted double (a crash is worse
    than not arriving), per episode, in [-2, 1]."""
    return round((summary["success_total"] - 2 * summary["collisions_total"]) / max(summary["of_total"], 1), 3)
