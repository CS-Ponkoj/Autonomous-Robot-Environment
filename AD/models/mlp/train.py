"""Train the MLP on the recorded examples, check its accuracy on the validation layouts,
benchmark it driving, and log the run.

    .venv\Scripts\python -m AD.models.mlp.train --run mlp1 --epochs 20"""

from __future__ import annotations

import argparse
import functools
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from robot_env import config as C

from ...common import benchmark, dataset, tracker
from ...common.features import ACTION_NAMES, LABEL_LEVEL, ODOMETRY, labels_of
from .model import MLPDriver, Net

HERE = Path(__file__).resolve().parent


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--head", choices=("discrete", "cont"), default="discrete", help="8 actions, or continuous (v, omega)")
    p.add_argument("--weighted", action="store_true", help="weight classes by inverse frequency (discrete head)")
    p.add_argument("--no-odometry", action="store_true", help="blank the measured speed and turn rate (copycat fix)")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--bench-device", default=None, help="device for benchmark workers (cpu saves RAM)")
    p.add_argument("--no-bench", action="store_true")
    p.add_argument("--final", action="store_true", help="also run the held-out test layouts (finished models only)")
    a = p.parse_args(argv)
    torch.manual_seed(0)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # owner: use the GPU wherever possible
    tr, va = dataset.load(dataset.TRAIN), dataset.load(dataset.VALID)
    if a.no_odometry:
        tr["x"][:, ODOMETRY] = 0.0
        va["x"][:, ODOMETRY] = 0.0
    arch = {"hidden": a.hidden, "layers": a.layers, "head": a.head}
    net = Net(**arch).to(dev)
    counts = np.bincount(tr["action"], minlength=len(ACTION_NAMES)).astype(np.float32)
    weight = torch.tensor(counts.sum() / np.maximum(counts, 1) / len(ACTION_NAMES)).to(dev) if a.weighted else None
    scale = torch.tensor(np.array(C.SPEED_LEVELS[LABEL_LEVEL], np.float32), device=dev)
    if a.head == "cont":  # imitate the recorded command itself, each axis as a share of its cap
        loss_fn = lambda out, y: nn.functional.mse_loss(out / scale, y / scale)  # noqa: E731
    else:
        loss_fn = nn.CrossEntropyLoss(weight=weight)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    X = torch.from_numpy(tr["x"]).to(dev)
    Y = torch.from_numpy(tr["cmd"] if a.head == "cont" else tr["action"]).to(dev)
    start = time.time()
    for _ in range(a.epochs):
        net.train()
        for idx in torch.randperm(len(X), device=dev).split(512):
            opt.zero_grad()
            loss_fn(net(X[idx]), Y[idx]).backward()
            opt.step()
    train_s = time.time() - start
    net.eval()
    with torch.no_grad():
        out = net(torch.from_numpy(va["x"]).to(dev)).cpu().numpy()
    if a.head == "cont":  # judged as the nearest action, plus the error of the command itself
        offline = dataset.accuracy(labels_of(out), va["action"])
        offline["mae_v"], offline["mae_omega"] = (round(float(m), 4) for m in np.abs(out - va["cmd"]).mean(0))
    else:
        offline = dataset.accuracy(out.argmax(1), va["action"])
    run_dir = HERE / "runs" / a.run
    run_dir.mkdir(parents=True, exist_ok=True)
    ck = run_dir / "model.pt"
    torch.save({"arch": arch, "state": {k: v.cpu() for k, v in net.state_dict().items()}, "no_odometry": a.no_odometry}, ck)
    entry = {"model": "mlp", "run": a.run, "kind": "finetune", "base": "scratch",
             "data": f"train seeds {dataset.TRAIN.start}-{dataset.TRAIN.stop - 1}, {len(Y)} examples",
             "params": {**arch, "epochs": a.epochs, "lr": a.lr, "weighted": a.weighted, "no_odometry": a.no_odometry}, "train_s": round(train_s, 1),
             "offline": offline, "notes": ("behavior cloning, " + ("continuous (v, omega)" if a.head == "cont" else "8 discrete actions")
                       + (", no speed inputs" if a.no_odometry else ""))}
    if not a.no_bench:
        summary = benchmark.run(functools.partial(MLPDriver, str(ck), a.bench_device), workers=a.workers)
        entry.update(bench=summary, score=benchmark.score(summary))
    if a.final:
        final = benchmark.run(functools.partial(MLPDriver, str(ck), a.bench_device), workers=a.workers, final=True)
        entry.update(final=final, final_score=benchmark.score(final))
    tracker.log(entry)
    print(a.run, "accuracy", offline["accuracy"], "score", entry.get("score"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
