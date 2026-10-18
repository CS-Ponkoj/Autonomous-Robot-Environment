"""Train the 1D-CNN policy by behavior cloning on the built-in driver's recorded decisions, keep
the epoch that does best on the validation layouts, benchmark it, and log the run.

    .venv\\Scripts\\python -m AD.models.cnn.train --head discrete --run bc_disc
    .venv\\Scripts\\python -m AD.models.cnn.train --head continuous --frames 4 --run bc_cont_f4

Discrete: cross-entropy on the 8 action labels (rare actions weighted up). Continuous: squared
error on the driver's (v, omega), each scaled by its cap. Checkpoints: AD/cnn/runs/<run>/."""

from __future__ import annotations

import argparse
import functools
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ...common import benchmark, dataset, tracker
from ...common.features import ACTION_NAMES, labels_of
from .policy import CNNDriver, CNNPolicy, V_CAP, W_CAP, params, save, stack

RUNS = tracker.AD / "models" / "cnn" / "runs"
LOCK_SCRIPT = tracker.AD.parent / "_private" / "machine_lock.sh"


def machine_lock(cmd: str, owner: str | None) -> None:
    """Take (waiting) or give back the shared machine lock around GPU training; owner None: no lock."""
    if owner:
        import shutil
        import subprocess
        git_bash = Path(r"C:\Program Files\Git\bin\bash.exe")  # a bare "bash" can find WSL's first
        bash = str(git_bash) if git_bash.exists() else shutil.which("bash") or "bash"
        subprocess.run([bash, str(LOCK_SCRIPT), cmd, owner], check=cmd == "acquire")


def outcome(seed: int) -> str | None:
    """A recorded episode's outcome: the stored field, or for the earliest recordings (none
    stored) success if it ended before the time limit, else timeout. None if not recorded."""
    f = dataset.DATA / f"seed{seed}.npz"
    if not f.exists():
        return None
    d = np.load(f)
    if "outcome" in d.files:
        return str(d["outcome"])
    return "success" if len(d["t"]) and d["t"][-1] < 129.9 else "timeout"


def reached(seeds) -> list[int]:
    """The recorded episodes in which the built-in driver reached the goal (it times out on
    some layouts; cloning those teaches the failure too)."""
    return [s for s in seeds if outcome(s) == "success"]


def data(seeds, frames: int, extra=None) -> dict:
    d = dataset.load(seeds)
    d["x"] = stack(d["x"], d["seed"], frames)
    if extra:  # aggregated (DAgger) states: already stacked, with the expert's labels
        for f in extra:
            e = np.load(f)
            d["x"] = np.concatenate([d["x"], e["x"]])
            d["cmd"] = np.concatenate([d["cmd"], e["cmd"]])
            d["action"] = np.concatenate([d["action"], labels_of(e["cmd"]).astype(np.int64)])
    return d


def evaluate(net: CNNPolicy, x: torch.Tensor, d: dict) -> dict:
    net.eval()
    with torch.no_grad():
        out = torch.cat([net(x[i:i + 8192]) for i in range(0, len(x), 8192)]).cpu()
    if net.head == "discrete":
        return dataset.accuracy(out.argmax(1).numpy(), d["action"])
    pred = out.numpy()
    r = dataset.accuracy(np.asarray(labels_of(pred)), d["action"])
    err = np.abs(pred - d["cmd"])
    r["mae_v"], r["mae_omega"] = round(float(err[:, 0].mean()), 4), round(float(err[:, 1].mean()), 4)
    return r


FRONT = slice(31, 41)  # lidar bins within 25 degrees of straight ahead (bin 0 starts at -180)
YIELD_RANGE = 0.4  # m: an obstacle this close ahead makes stop / back_up the lesson to learn


def yield_weights(d: dict, frames: int, weight: float) -> np.ndarray:
    """Per-example loss weights: `weight` for stop or back_up labels with something within
    YIELD_RANGE ahead (in the latest frame), 1 elsewhere."""
    from robot_env import config as C
    from ...common.features import FEATURES
    latest = d["x"][:, (frames - 1) * FEATURES:]
    near = latest[:, FRONT].min(1) * C.LIDAR_RANGE < YIELD_RANGE
    yielding = np.isin(d["action"], [ACTION_NAMES.index("stop"), ACTION_NAMES.index("back_up")])
    return np.where(near & yielding, weight, 1.0).astype(np.float32)


def fit(net: CNNPolicy, tr: dict, va: dict, epochs: int, batch: int, lr: float, device: str,
        yield_weight: float = 1.0) -> tuple[dict, int]:
    """Train; return the validation result of the best epoch (its weights are left in net)."""
    net.to(device)
    xt, xv = torch.from_numpy(tr["x"]).to(device), torch.from_numpy(va["x"]).to(device)
    if net.head == "discrete":
        yt = torch.from_numpy(tr["action"]).to(device)
        freq = np.bincount(tr["action"], minlength=len(ACTION_NAMES)) + 1.0
        wts = torch.tensor((freq.sum() / freq) ** 0.5, dtype=torch.float32)
        ce = nn.CrossEntropyLoss(weight=(wts / wts.mean()).to(device), reduction="none")
        ex = torch.from_numpy(yield_weights(tr, net.frames, yield_weight)).to(device)
        idx_of = {}  # the batch's example weights are looked up by the indices drawn

        def loss_fn(out, y):
            return (ce(out, y) * ex[idx_of["i"]]).mean()
    else:
        yt = torch.from_numpy(tr["cmd"]).to(device)
        scale = torch.tensor([V_CAP, W_CAP], device=device)
        loss_fn = lambda out, y: (((out - y) / scale) ** 2).mean()  # noqa: E731
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    best, best_ep, best_state = None, -1, None
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(xt), device=device)
        for i in range(0, len(xt), batch):
            idx = perm[i:i + batch]
            if net.head == "discrete":
                idx_of["i"] = idx
            loss = loss_fn(net(xt[idx]), yt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
        r = evaluate(net, xv, va)
        key = r["accuracy"] if net.head == "discrete" else -(r["mae_v"] / V_CAP + r["mae_omega"] / W_CAP)
        if best is None or key > best[0]:
            best, best_ep = (key, r), ep
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        print(f"epoch {ep + 1}/{epochs} loss {loss.item():.4f} valid {r['accuracy']:.3f}"
              + (f" mae v {r['mae_v']:.3f} w {r['mae_omega']:.3f}" if net.head == "continuous" else ""), flush=True)
    net.load_state_dict(best_state)
    return best[1], best_ep


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--head", choices=("discrete", "continuous"), default="discrete")
    p.add_argument("--frames", type=int, default=1)
    p.add_argument("--width", type=int, default=32)
    p.add_argument("--no-odometry", action="store_true", help="hide the measured v and omega (against the copycat shortcut)")
    p.add_argument("--yield-weight", type=float, default=1.0, help="loss weight of stop/back_up labels with an obstacle within 0.4 m ahead")
    p.add_argument("--run", required=True)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=1024)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--train", default=None, help="seed range a-b (default: dataset.TRAIN)")
    p.add_argument("--valid", default=None, help="seed range a-b (default: dataset.VALID)")
    p.add_argument("--extra", nargs="*", default=None, help="aggregated DAgger state files")
    p.add_argument("--reached-only", action="store_true", help="train only on episodes that reached the goal")
    p.add_argument("--init", default=None, help="start from this checkpoint (fine-tune)")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--no-bench", action="store_true")
    p.add_argument("--bench-device", default=None, choices=("cpu", "cuda"),
                   help="where the benchmark workers run the net (default: the GPU if any; cpu saves host RAM per worker)")
    p.add_argument("--lock", default=None, help="hold the machine lock as this owner while training (not while benchmarking)")
    p.add_argument("--notes", default="")
    a = p.parse_args(argv)
    rng = lambda s, d: d if s is None else range(int(s.split("-")[0]), int(s.split("-")[1]) + 1)  # noqa: E731
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_seeds = rng(a.train, dataset.TRAIN)
    tr = data(reached(train_seeds) if a.reached_only else train_seeds, a.frames, a.extra)
    va = data(rng(a.valid, dataset.VALID), a.frames)
    net = CNNPolicy(a.head, a.frames, a.width, odometry=not a.no_odometry)
    if a.init:
        from .policy import load
        net = load(a.init)
    machine_lock("acquire", a.lock)
    try:
        t0 = time.time()
        offline, best_ep = fit(net, tr, va, a.epochs, a.batch, a.lr, device, a.yield_weight)
        train_s = time.time() - t0
    finally:
        machine_lock("release", a.lock)
    path = RUNS / a.run / "model.pt"
    save(net.cpu(), path, {"run": a.run})
    print("offline", offline, "best epoch", best_ep + 1, f"train {train_s:.0f} s", flush=True)
    entry = {"model": "cnn", "run": a.run, "kind": "finetune", "base": Path(a.init).parent.name if a.init else "scratch",
             "data": f"train {len(tr['x'])} examples" + (" (episodes that reached the goal)" if a.reached_only else "") + (f" (+{len(a.extra)} DAgger files)" if a.extra else ""),
             "params": {"head": a.head, "frames": a.frames, "width": a.width, "odometry": not a.no_odometry, "yield_weight": a.yield_weight, "epochs": a.epochs, "best_epoch": best_ep + 1,
                        "lr": a.lr, "batch": a.batch, "weights": params(net)},
             "train_s": round(train_s, 1), "offline": offline, "notes": a.notes}
    if not a.no_bench:
        summary = benchmark.run(functools.partial(CNNDriver, str(path), device=a.bench_device), workers=a.workers)
        entry["bench"], entry["score"] = summary, benchmark.score(summary)
        print({k: summary[k] for k in ("no_cats", "4_cats", "latency_ms")}, "score", entry["score"], flush=True)
    tracker.log(entry)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
