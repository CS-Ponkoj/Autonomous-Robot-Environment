"""Train the map CNN by behavior cloning on the built-in driver's recorded decisions (the maps
rebuilt by replaying each recording's observations, maps.py), keep the epoch that does best on
the validation layouts, benchmark it on CPU workers, and log the run.

    .venv\\Scripts\\python -m AD.models.mapcnn.train --run bc --lock ARE2-gpu

Loss: squared error on the driver's (v, omega), each scaled by its cap. The rebuilt maps are
cached per seed in AD/models/mapcnn/runs/_maps/ (rebuilt when maps.py's settings change)."""

from __future__ import annotations

import argparse
import functools
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

from ...common import benchmark, dataset, tracker
from ...common.features import labels_of
from . import maps
from .policy import MapDriver, MapPolicy, V_CAP, W_CAP, params, save

RUNS = tracker.AD / "models" / "mapcnn" / "runs"
CACHE = RUNS / "_maps"
GUARD = tracker.AD / "private" / "guard_state.json"
LOCK_SCRIPT = tracker.AD.parent / "_private" / "machine_lock.sh"
MAP_KEY = json.dumps([maps.CELLS, maps.RES, maps.CHANNELS, maps.OCC_DECAY, maps.FREE_DECAY, maps.CLEAR])
MIN_COMMIT_GB = 10.0


def machine_lock(cmd: str, owner: str | None) -> None:
    if owner:
        import shutil
        import subprocess
        git_bash = Path(r"C:\Program Files\Git\bin\bash.exe")  # a bare "bash" can find WSL's first
        bash = str(git_bash) if git_bash.exists() else shutil.which("bash") or "bash"
        subprocess.run([bash, str(LOCK_SCRIPT), cmd, owner], check=cmd == "acquire")


def check_memory() -> None:
    """Start a heavy job only with enough commit memory free (the shared guard's last reading)."""
    try:
        free = json.loads(GUARD.read_text(encoding="utf-8"))["commit_free_gb"]
    except (OSError, ValueError, KeyError):
        return
    if free < MIN_COMMIT_GB:
        raise SystemExit(f"only {free} GB commit free (need {MIN_COMMIT_GB}); try again later")


def _build(seed: int) -> int:
    out = CACHE / f"seed{seed}.npz"
    if out.exists() and str(np.load(out)["key"]) == MAP_KEY:
        return 0
    d = np.load(dataset.DATA / f"seed{seed}.npz")
    m = maps.replay(d["x"], d["t"])
    tmp = out.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, maps=m, key=MAP_KEY)
    tmp.replace(out)
    return len(m)


def build(seeds, workers: int) -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    todo = [s for s in seeds if (dataset.DATA / f"seed{s}.npz").exists()]
    t0 = time.time()
    with ProcessPoolExecutor(workers) as pool:
        n = sum(pool.map(_build, todo, chunksize=8))
    if n:
        print(f"built {n} maps for {len(todo)} seeds in {time.time() - t0:.0f} s", flush=True)


def data(seeds, extra=None) -> dict:
    """x, cmd, action, seed (as dataset.load) and the maps, row for row; plus DAgger state files."""
    d = dataset.load(seeds)
    order = list(dict.fromkeys(d["seed"].tolist()))  # the seeds in load order (empty ones skipped)
    d["maps"] = np.concatenate([np.load(CACHE / f"seed{s}.npz")["maps"] for s in order])
    assert len(d["maps"]) == len(d["x"]), "maps and recordings differ in length"
    for f in extra or []:
        e = np.load(f)
        d["x"], d["maps"] = np.concatenate([d["x"], e["x"]]), np.concatenate([d["maps"], e["maps"]])
        d["cmd"] = np.concatenate([d["cmd"], e["cmd"]])
        d["action"] = np.concatenate([d["action"], labels_of(e["cmd"]).astype(np.int64)])
    return d


def evaluate(net, x, m, d) -> dict:
    net.eval()
    with torch.no_grad():
        pred = torch.cat([net(x[i:i + 4096], m[i:i + 4096]) for i in range(0, len(x), 4096)]).cpu().numpy()
    r = dataset.accuracy(np.asarray(labels_of(pred)), d["action"])
    err = np.abs(pred - d["cmd"])
    r["mae_v"], r["mae_omega"] = round(float(err[:, 0].mean()), 4), round(float(err[:, 1].mean()), 4)
    return r


def fit(net, tr, va, epochs: int, batch: int, lr: float, device: str) -> tuple[dict, int]:
    net.to(device)
    xt, mt, yt = (torch.from_numpy(tr[k]).to(device) for k in ("x", "maps", "cmd"))
    xv, mv = torch.from_numpy(va["x"]).to(device), torch.from_numpy(va["maps"]).to(device)
    scale = torch.tensor([V_CAP, W_CAP], device=device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    best, best_ep, best_state = None, -1, None
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(xt), device=device)
        for i in range(0, len(xt), batch):
            idx = perm[i:i + batch]
            loss = (((net(xt[idx], mt[idx]) - yt[idx]) / scale) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
        r = evaluate(net, xv, mv, va)
        key = -(r["mae_v"] / V_CAP + r["mae_omega"] / W_CAP)
        if best is None or key > best[0]:
            best, best_ep = (key, r), ep
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        print(f"epoch {ep + 1}/{epochs} loss {loss.item():.4f} valid {r['accuracy']:.3f} "
              f"mae v {r['mae_v']:.3f} w {r['mae_omega']:.3f}", flush=True)
    net.load_state_dict(best_state)
    return best[1], best_ep


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--run", required=True)
    p.add_argument("--width", type=int, default=32)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch", type=int, default=1024)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--train", default=None, help="seed range a-b (default: dataset.TRAIN)")
    p.add_argument("--valid", default=None, help="seed range a-b (default: dataset.VALID)")
    p.add_argument("--extra", nargs="*", default=None, help="DAgger state files (x, maps, cmd)")
    p.add_argument("--init", default=None, help="start from this checkpoint (fine-tune)")
    p.add_argument("--map-workers", type=int, default=6, help="processes rebuilding the maps")
    p.add_argument("--workers", type=int, default=4, help="benchmark processes (CPU)")
    p.add_argument("--no-bench", action="store_true")
    p.add_argument("--lock", default=None, help="hold the machine lock as this owner while training")
    p.add_argument("--notes", default="")
    a = p.parse_args(argv)
    check_memory()
    rng = lambda s, d: d if s is None else range(int(s.split("-")[0]), int(s.split("-")[1]) + 1)  # noqa: E731
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_seeds, valid_seeds = rng(a.train, dataset.TRAIN), rng(a.valid, dataset.VALID)
    build(list(train_seeds) + list(valid_seeds), a.map_workers)
    tr, va = data(train_seeds, a.extra), data(valid_seeds)
    net = MapPolicy(a.width)
    if a.init:
        from .policy import load
        net = load(a.init)
    machine_lock("acquire", a.lock)
    try:
        t0 = time.time()
        offline, best_ep = fit(net, tr, va, a.epochs, a.batch, a.lr, device)
        train_s = time.time() - t0
    finally:
        machine_lock("release", a.lock)
    n_train = len(tr["x"])
    tr = va = None  # free the data before the benchmark workers start
    path = RUNS / a.run / "model.pt"
    save(net.cpu(), path, {"run": a.run})
    print("offline", offline, "best epoch", best_ep + 1, f"train {train_s:.0f} s", flush=True)
    entry = {"model": "mapcnn", "run": a.run, "kind": "finetune", "base": Path(a.init).parent.name if a.init else "scratch",
             "data": f"train {n_train} examples" + (f" (+{len(a.extra)} DAgger files)" if a.extra else ""),
             "params": {"width": a.width, "epochs": a.epochs, "best_epoch": best_ep + 1, "lr": a.lr, "batch": a.batch,
                        "map": {"cells": maps.CELLS, "res_m": maps.RES, "channels": maps.CHANNELS,
                                "occ_decay": maps.OCC_DECAY, "free_decay": maps.FREE_DECAY, "clear": maps.CLEAR},
                        "odometry_input": False, "weights": params(net)},
             "train_s": round(train_s, 1), "offline": offline, "notes": a.notes}
    if not a.no_bench:
        summary = benchmark.run(functools.partial(MapDriver, str(path)), workers=a.workers)
        entry["bench"], entry["score"] = summary, benchmark.score(summary)
        print({k: summary[k] for k in ("no_cats", "4_cats", "latency_ms")}, "score", entry["score"], flush=True)
    tracker.log(entry)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
