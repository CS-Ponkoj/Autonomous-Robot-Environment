"""DAgger with the route input for any family (hybrid CNN, cfc, dt, mlp, mapcnn): the policy
drives training layouts with its route source, an expert planner labels every state it reaches,
and the policy is fine-tuned on all rounds' episodes, benchmarked and logged (<run>_r<k>).

    .venv\\Scripts\\python -m AD.models.hybrid.route_dagger --family cfc --base cfc/dagger_cfc_r3 --run route_cfc --rounds 2
    .venv\\Scripts\\python -m AD.models.hybrid.route_dagger --family hybrid --init hppo_c3 --run hybrid_cats --cats 3 4 4
    .venv\\Scripts\\python -m AD.models.hybrid.route_dagger --family mlp --base mlp/mlp_cont --run route_mlp --route map

Round 1 starts from --base (a trained policy of the family without route input, widened with
zero weights on the route values) or from --init (a route run of the family to continue). Each
round's episodes are stored whole (AD/models/hybrid/routekit.py Episodes: the 78 features, times,
route values, the expert's commands), so sequence models train on sequences and the map CNN
rebuilds its maps; --reuse-states adds earlier rounds' episode files. --cats cycles cat counts
over a round's episodes (default seed % 5, as recorded); cats are the known weak spot.

The expert must plan on the same route the policy reads, or its labels steer against the route
input: --expert map (the default with --route map) labels with the map planner on the route
source's saved map and drift correction; --expert online with the built-in planner. Each episode
file's .json records its expert; files marked label_inconsistent are refused.

The recordings have no route values (they keep only features.vector), so route policies learn
from DAgger episodes. Run from the main checkout (its robot_env: the compiled planner)."""

from __future__ import annotations

import argparse
import functools
import importlib
import json
import random
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from ...common import benchmark, dataset, tracker
from ...common.features import FEATURES
from .route import reset_driver, route_sources, route_spec
from .routekit import Episodes, source_spec, spec_of, split_episodes

FAMILIES = ("hybrid", "cfc", "dt", "mlp", "mapcnn")
EXPERTS = ("online", "map")


def adapter(family: str):
    return importlib.import_module(f"AD.models.{family}.routed")


def runs_dir(family: str) -> Path:
    return tracker.AD / "models" / family / "runs"


def make_expert(kind: str, policy):
    """The labelling planner: "online" the built-in planner; "map" the map planner on the saved
    map and drift correction of the policy's own route source (so labels follow the route it reads)."""
    from robot_env import config as C
    from robot_env.baseline import BaselineDriver

    from ...common.features import LABEL_LEVEL
    if kind == "online":
        return BaselineDriver(C.SPEED_LEVELS[LABEL_LEVEL])
    spec = route_spec(policy.source)
    if spec.get("name") not in ("map", "map2"):
        raise ValueError(f"--expert map needs a map route source, not {spec.get('name')!r}")
    from ..mapplan.driver import MapDriver
    from ..mapplan.savedmap import DEFAULT
    # the same planner settings as the route source (map2: off-map obstacles forgotten once seen gone)
    return MapDriver(DEFAULT.parent.parent / spec["map"] / "map.npz", C.SPEED_LEVELS[LABEL_LEVEL], match=spec["match"],
                     nonmap_ttl=spec.get("nonmap_ttl"), nonmap_clear=bool(spec.get("nonmap_clear", False)))


def rollout(args) -> dict:
    """One episode: the route policy drives (the expert's command instead with probability beta),
    the expert labels every state."""
    family, ckpt, seed, beta, cats, route, expert_kind = args
    from robot_env import config as C
    from robot_env.layout import RoomMap
    from robot_env.system import RobotSystem

    from ...common.features import vector

    policy = adapter(family).Driver(ckpt, "cpu", route)
    expert = make_expert(expert_kind, policy)
    system = RobotSystem(cats=seed % 5 if cats is None else cats, cat_seed=seed)
    task = RoomMap(system.sim.model).sample_task(seed)
    system.reset(*task.start, task.goal)
    reset_driver(policy, task.start)
    reset_driver(expert, task.start)
    coin = random.Random(seed * 7919 + int(beta * 1000))
    X, T, R, V, outcome = [], [], [], [], "timeout"
    while system.time < C.EPISODE_TIME_LIMIT - 1e-9:
        obs = system.observe()
        if obs.goal_distance <= C.GOAL_RADIUS:
            outcome = "success"
            break
        label = expert.decide(obs)
        mine = policy.decide(obs)
        cmd = label.command if label else None
        X.append(vector(obs))
        T.append(obs.time)
        R.append(policy.last_route)
        V.append((cmd.v, cmd.omega) if cmd else (0.0, 0.0))
        system.apply(label if (label is not None and coin.random() < beta) else mine)
        system.advance(C.DECISION_PERIOD)
        if system.collisions:
            outcome = "collision"
            break
    system.close()
    # fixed shapes even for an episode that ends at its first observation (no decisions)
    return {"seed": seed, "outcome": outcome, "x": np.array(X, np.float32).reshape(-1, FEATURES),
            "t": np.array(T, np.float32).reshape(-1), "route": np.array(R, np.float32).reshape(-1, 4),
            "cmd": np.array(V, np.float32).reshape(-1, 2)}


def collect(family, ckpt, seeds, beta, workers, out: Path, cats=None, route=None, expert="online") -> dict:
    jobs = [(family, ckpt, s, beta, cats[i % len(cats)] if cats else None, route, expert) for i, s in enumerate(seeds)]
    with ProcessPoolExecutor(workers, max_tasks_per_child=1) as pool:  # a fresh process per episode
        eps = list(pool.map(rollout, jobs))
    Episodes.save(out, eps)
    return {"states": int(sum(len(e["x"]) for e in eps)), "cats": cats, "route": route, "expert": expert,
            "outcomes": {k: sum(e["outcome"] == k for e in eps) for k in ("success", "collision", "timeout")}}


def check_labels(files, expert: str) -> None:
    """Refuse episode files marked label_inconsistent, or labelled by another expert than this run's
    (files from before the expert was recorded were labelled online)."""
    for f in files:
        meta = Path(f).with_suffix(".json")
        info = json.loads(meta.read_text(encoding="utf-8")) if meta.exists() else {}
        if info.get("label_inconsistent"):
            raise SystemExit(f"{f} is marked label_inconsistent: {info['label_inconsistent']}")
        if info.get("expert", "online") != expert:
            raise SystemExit(f"{f} was labelled by the {info.get('expert', 'online')!r} expert, this run uses {expert!r}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--family", required=True, choices=FAMILIES)
    p.add_argument("--base", default=None, help="<model>/<run>: a trained policy of the family without route input")
    p.add_argument("--init", default=None, help="a route run of the family to continue (instead of --base)")
    p.add_argument("--run", required=True)
    p.add_argument("--rounds", type=int, default=2)
    p.add_argument("--start-round", type=int, default=1,
                   help="resume a stopped run at round k: continue from <run>_r<k-1> with rounds 1..k-1's episodes")
    p.add_argument("--episodes", type=int, default=60)
    p.add_argument("--beta", type=float, default=0.5, help="expert share in round 1, halved each round")
    p.add_argument("--cats", type=int, nargs="*", default=None, help="cat counts cycled over each round's episodes")
    p.add_argument("--route", default=None,
                   help=f"route source ({', '.join(route_sources())}); default: --init's own, else online")
    p.add_argument("--allow-route-change", action="store_true",
                   help="allow --route to differ from the --init checkpoint's source (retrain on another source)")
    p.add_argument("--expert", default=None, choices=EXPERTS,
                   help="labelling planner: map (default with --route map: the route source's saved map) or online")
    p.add_argument("--reuse-states", nargs="*", default=[], help="earlier episode files (.npz) to train on too")
    p.add_argument("--epochs", type=int, default=15, help="epochs (cnn, mlp, mapcnn)")
    p.add_argument("--steps", type=int, default=2000, help="optimizer steps (cfc, dt)")
    p.add_argument("--batch", type=int, default=128, help="windows per step (cfc, dt)")
    p.add_argument("--lr", type=float, default=None, help="default: 5e-4 (cnn, mlp, mapcnn), 3e-4 (cfc, dt)")
    p.add_argument("--workers", type=int, default=2, help="rollout and benchmark processes")
    p.add_argument("--first-seed", type=int, default=300, help="offset into the training seeds")
    p.add_argument("--no-bench", action="store_true")
    p.add_argument("--no-log", action="store_true", help="a smoke run: no log entry (not a model)")
    a = p.parse_args(argv)
    fam = adapter(a.family)
    runs = runs_dir(a.family)
    if a.start_round > 1:  # resume: the previous round's model is the start, its episodes are kept
        if a.base or a.init:
            p.error("--start-round continues <run>_r<k-1>; leave out --base and --init")
        a.init = f"{a.run}_r{a.start_round - 1}"
        if not (runs / a.init / "model.pt").exists():
            p.error(f"no {a.init} to resume from")
    elif bool(a.base) == bool(a.init):
        p.error("give exactly one of --base and --init")
    if a.init:  # the source the run was trained with, unless a change is asked for explicitly
        own = spec_of(torch.load(runs / a.init / "model.pt", map_location="cpu", weights_only=False))["name"]
        if a.route is None:
            a.route = own
        elif a.route != own and not a.allow_route_change:
            p.error(f"--init {a.init} was trained with route source {own!r}, not {a.route!r} "
                    "(add --allow-route-change to retrain it on another source)")
    a.route = a.route or "online"
    a.expert = a.expert or ("map" if a.route in ("map", "map2") else "online")
    if a.expert == "map" and a.route not in ("map", "map2"):
        p.error("--expert map needs --route map (the expert plans on the route source's saved map)")
    spec = source_spec(a.route)
    torch.manual_seed(0)
    fit_args = SimpleNamespace(epochs=a.epochs, steps=a.steps, batch=a.batch, device="cuda" if torch.cuda.is_available() else "cpu",
                               lr=a.lr or (3e-4 if a.family in ("cfc", "dt") else 5e-4))
    if a.init:
        ckpt = str(runs / a.init / "model.pt")
    else:
        m, r = a.base.split("/")
        start = runs / f"{a.run}_init" / "model.pt"  # the widened base (no training: not a model of its own)
        fam.save(fam.widen_from(tracker.AD / "models" / m / "runs" / r / "model.pt"), start, spec, {"run": start.parent.name})
        ckpt = str(start)
    pool = list(dataset.TRAIN)
    earlier = [str(runs / a.run / f"episodes_r{j}.npz") for j in range(1, a.start_round)]
    missing = [f for f in earlier if not Path(f).exists()]
    if missing:
        p.error(f"--start-round {a.start_round} needs the earlier rounds' episodes: {missing}")
    files, beta = list(a.reuse_states) + earlier, a.beta / 2 ** (a.start_round - 1)
    check_labels(files, a.expert)
    for k in range(a.start_round, a.rounds + 1):
        seeds = [pool[(a.first_seed + (k - 1) * a.episodes + i) % len(pool)] for i in range(a.episodes)]
        out = runs / a.run / f"episodes_r{k}.npz"
        meta = out.with_suffix(".json")
        if out.exists() and meta.exists():  # collected before (a rerun): train from it
            check_labels([out], a.expert)
            info = json.loads(meta.read_text(encoding="utf-8"))
        else:
            t0 = time.time()
            info = collect(a.family, ckpt, seeds, beta, a.workers, out, a.cats, a.route, a.expert)
            meta.write_text(json.dumps(info), encoding="utf-8")
            print(f"round {k}: collected {info} in {time.time() - t0:.0f} s (beta {beta:.2f})", flush=True)
        files.append(str(out))
        train, valid = split_episodes(Episodes.load(files))  # the same held-out episodes every round
        net = fam.load(ckpt)
        t0 = time.time()
        net, offline = fam.fit(net, train, valid, fit_args)
        train_s = time.time() - t0
        run = f"{a.run}_r{k}"
        path = runs / run / "model.pt"
        fam.save(net, path, spec, {"run": run})
        print("offline", offline, flush=True)
        entry = {"model": a.family, "run": run, "kind": "finetune", "base": (a.init or a.base) if k == 1 else f"{a.run}_r{k - 1}",
                 "data": f"{sum(len(e['x']) for e in train)} route DAgger states ({len(train)} episodes, {len(files)} files)",
                 "params": {"route": spec, "expert": a.expert, "cats": a.cats, "beta": beta, "episodes": a.episodes, "epochs": a.epochs,
                            "steps": a.steps, "lr": fit_args.lr, "weights": fam.params(net)},
                 "train_s": round(train_s, 1), "offline": offline,
                 "notes": f"route DAgger round {k} ({a.family}, source {spec['name']}, expert {a.expert}): +{info['states']} states "
                          f"(policy episodes {info['outcomes']}, beta {beta:.2f}, cats {a.cats or 'seed % 5'})"}
        if not a.no_bench:
            summary = benchmark.run(functools.partial(fam.Driver, str(path), "cpu", a.route), workers=a.workers)
            entry["bench"], entry["score"] = summary, benchmark.score(summary)
            print({kk: summary[kk] for kk in ("no_cats", "4_cats", "latency_ms")}, "score", entry["score"], flush=True)
        if not a.no_log:
            tracker.log(entry)
        ckpt, beta = str(path), beta / 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
