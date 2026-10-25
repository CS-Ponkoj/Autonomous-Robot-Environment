"""DAgger for the hybrid policy (the CNN plus the built-in planner's route as 4 inputs).

    .venv\\Scripts\\python -m AD.models.hybrid.dagger --base cnn/dagger_cont_r3 --run hybrid_dagger --rounds 2

The recorded expert data has no route values (the recordings keep only features.vector), so the
hybrid learns from its own DAgger rollouts only: each round the hybrid drives training layouts, the
route sensor reads every state, the built-in driver labels it, and the hybrid is fine-tuned on all
rounds' states so far (10% held out at random for model selection). Round 1 starts from the cloned
CNN widened with zero weights on the route inputs (it drives exactly as dagger_cont_r3 then). Each
round is benchmarked (CPU drivers) and logged as its own run (<run>_r<k>)."""

from __future__ import annotations

import argparse
import functools
import json
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch

from ...common import benchmark, dataset, tracker
from ...common.features import labels_of
from ..cnn.dagger import rollout
from ..cnn.policy import load as load_cnn, params, save as save_cnn
from ..cnn.train import fit
from .policy import HybridDriver, widen

RUNS = tracker.AD / "models" / "hybrid" / "runs"


def collect(ckpt: str, seeds: list[int], beta: float, workers: int, out) -> dict:
    with ProcessPoolExecutor(workers, max_tasks_per_child=1) as pool:  # a fresh process per episode
        eps = list(pool.map(rollout, [(ckpt, s, beta, "hybrid") for s in seeds]))
    np.savez_compressed(out, x=np.concatenate([e["x"] for e in eps]), cmd=np.concatenate([e["cmd"] for e in eps]))
    return {"states": int(sum(len(e["x"]) for e in eps)),
            "outcomes": {k: sum(e["outcome"] == k for e in eps) for k in ("success", "collision", "timeout")}}


def states(files: list[str], rng: np.random.Generator) -> tuple[dict, dict]:
    x = np.concatenate([np.load(f)["x"] for f in files]).astype(np.float32)
    cmd = np.concatenate([np.load(f)["cmd"] for f in files]).astype(np.float32)
    idx = rng.permutation(len(x))
    cut = len(x) // 10
    split = lambda i: {"x": x[i], "cmd": cmd[i], "action": np.asarray(labels_of(cmd[i]))}  # noqa: E731
    return split(idx[cut:]), split(idx[:cut])


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--base", default="cnn/dagger_cont_r3", help="<model>/<run> of the continuous CNN to widen")
    p.add_argument("--run", default="hybrid_dagger")
    p.add_argument("--rounds", type=int, default=2)
    p.add_argument("--episodes", type=int, default=60)
    p.add_argument("--beta", type=float, default=0.5, help="expert share in round 1, halved each round")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--first-seed", type=int, default=120, help="offset into the training seeds")
    a = p.parse_args(argv)
    rng = np.random.default_rng(0)
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    m, r = a.base.split("/")
    init = RUNS / f"{a.run}_init" / "model.pt"  # the widened start (not a model of its own: no training)
    if not init.exists():
        save_cnn(widen(load_cnn(tracker.AD / "models" / m / "runs" / r / "model.pt")), init, {"run": init.parent.name})
    pool = list(dataset.TRAIN)
    files, ckpt, beta = [], str(init), a.beta
    for k in range(1, a.rounds + 1):
        seeds = [pool[(a.first_seed + (k - 1) * a.episodes + i) % len(pool)] for i in range(a.episodes)]
        out = RUNS / a.run / f"states_r{k}.npz"
        out.parent.mkdir(parents=True, exist_ok=True)
        meta = out.with_suffix(".json")
        if out.exists() and meta.exists():
            info = json.loads(meta.read_text(encoding="utf-8"))
        else:
            t0 = time.time()
            info = collect(ckpt, seeds, beta, a.workers, out)
            meta.write_text(json.dumps(info), encoding="utf-8")
            print(f"round {k}: collected {info} in {time.time() - t0:.0f} s (beta {beta:.2f})", flush=True)
        files.append(str(out))
        tr, va = states(files, rng)
        net = load_cnn(ckpt)
        t0 = time.time()
        offline, best_ep = fit(net, tr, va, a.epochs, 1024, a.lr, device)
        train_s = time.time() - t0
        run = f"{a.run}_r{k}"
        path = RUNS / run / "model.pt"
        save_cnn(net.cpu(), path, {"run": run})
        print("offline", offline, "best epoch", best_ep + 1, flush=True)
        summary = benchmark.run(functools.partial(HybridDriver, str(path), "cpu"), workers=a.workers)
        entry = tracker.log({
            "model": "hybrid", "run": run, "kind": "finetune", "base": a.base if k == 1 else f"{a.run}_r{k - 1}",
            "data": f"{len(tr['x'])} hybrid DAgger states from {k} rounds (route sensor inputs; 10% held out)",
            "params": {"epochs": a.epochs, "best_epoch": best_ep + 1, "lr": a.lr, "weights": params(net), "extra": net.extra,
                       "beta": beta, "episodes": a.episodes},
            "train_s": round(train_s, 1), "offline": offline, "bench": summary, "score": benchmark.score(summary),
            "notes": f"hybrid DAgger round {k}: +{info['states']} states (policy episodes {info['outcomes']}, beta {beta:.2f})"})
        print({kk: summary[kk] for kk in ("no_cats", "4_cats", "latency_ms")}, "score", entry["score"], flush=True)
        ckpt, beta = str(path), beta / 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
