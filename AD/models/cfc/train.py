"""Train the CfC policy by behavior cloning on the recorded episodes, benchmark and log it.

    .venv\\Scripts\\python -m AD.models.cfc.train --run cfc_bc

Training windows: WINDOW steps from an episode, the first BURN_IN of them only to warm the state up
(no loss there), so the memory is trained as the driver uses it: carried along an episode. The
target is the built-in driver's (v, omega), each scaled by its cap. The epoch with the lowest
validation error is kept (teacher-forced over whole validation episodes)."""

from __future__ import annotations

import argparse
import functools
import time

import numpy as np
import torch

from ...common import benchmark, dataset, tracker
from ...common.features import labels_of
from .model import V_CAP, W_CAP, CfCDriver, CfCPolicy, params, save

RUNS = tracker.AD / "models" / "cfc" / "runs"
WINDOW, BURN_IN = 48, 16


def episodes(seeds, extra=None) -> list[dict]:
    """The recorded episodes, plus any DAgger files (episodes the policy drove, labelled by the
    built-in driver: x, cmd and the episode lengths)."""
    d = dataset.load(seeds)
    eps = [{"x": d["x"][d["seed"] == s], "cmd": d["cmd"][d["seed"] == s], "action": d["action"][d["seed"] == s]}
           for s in np.unique(d["seed"])]
    for f in extra or []:
        e = np.load(f)
        for x, c in zip(np.split(e["x"], np.cumsum(e["lens"])[:-1]), np.split(e["cmd"], np.cumsum(e["lens"])[:-1])):
            if len(x):
                eps.append({"x": x, "cmd": c, "action": np.asarray(labels_of(c))})
    return eps


def windows(eps: list[dict], n: int, rng: np.random.Generator):
    lens = np.array([len(e["x"]) for e in eps])
    pick = rng.choice(len(eps), n, p=lens / lens.sum())
    X = np.zeros((n, WINDOW, eps[0]["x"].shape[1]), np.float32)
    Y = np.zeros((n, WINDOW, 2), np.float32)
    M = np.zeros((n, WINDOW), np.float32)
    for i, j in enumerate(pick):
        e = eps[j]
        end = int(rng.integers(1, len(e["x"]) + 1))
        lo = max(0, end - WINDOW)
        m = end - lo
        # a window holding the episode's start begins there (the state starts at zero, as when
        # driving; padding only after it, without loss); one from the middle warms up first
        X[i, :m], Y[i, :m], M[i, :m] = e["x"][lo:end], e["cmd"][lo:end], 1.0
        if lo > 0:
            M[i, :BURN_IN] = 0.0
    return X, Y, M


def evaluate(net: CfCPolicy, eps: list[dict], device: str) -> dict:
    """Teacher-forced over whole validation episodes (the state carried from the start)."""
    net.eval()
    preds, cmds, acts = [], [], []
    with torch.no_grad():
        for e in eps:
            out, _ = net(torch.from_numpy(e["x"])[None].to(device))
            preds.append(out[0].cpu().numpy()), cmds.append(e["cmd"]), acts.append(e["action"])
    p, c = np.concatenate(preds), np.concatenate(cmds)
    r = dataset.accuracy(np.asarray(labels_of(p)), np.concatenate(acts))
    r["mae_v"], r["mae_omega"] = round(float(np.abs(p - c)[:, 0].mean()), 4), round(float(np.abs(p - c)[:, 1].mean()), 4)
    return r


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--run", required=True)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--no-bench", action="store_true")
    p.add_argument("--extra", nargs="*", default=None, help="DAgger episode files (AD/models/cfc/dagger.py)")
    p.add_argument("--init", default=None, help="start from this checkpoint (fine-tune)")
    p.add_argument("--notes", default="")
    a = p.parse_args(argv)
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tr, va = episodes(dataset.TRAIN, a.extra), episodes(dataset.VALID)
    if a.init:
        from .model import load
        net = load(a.init).to(device)
    else:
        net = CfCPolicy(a.hidden).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)
    scale = torch.tensor([V_CAP, W_CAP], device=device)
    t0, best, best_state, best_step = time.time(), None, None, 0
    for step in range(1, a.steps + 1):
        net.train()
        X, Y, M = (torch.from_numpy(v).to(device) for v in windows(tr, a.batch, rng))
        out, _ = net(X)
        loss = ((((out - Y) / scale) ** 2).sum(-1) * M).sum() / M.sum()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 1000 == 0 or step == a.steps:
            r = evaluate(net, va, device)
            key = -(r["mae_v"] / V_CAP + r["mae_omega"] / W_CAP)
            if best is None or key > best[0]:
                best, best_step = (key, r), step
                best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            print(f"step {step} loss {loss.item():.4f} valid acc {r['accuracy']:.3f} mae v {r['mae_v']:.3f} w {r['mae_omega']:.3f}",
                  flush=True)
    train_s = time.time() - t0
    net.load_state_dict(best_state)
    path = RUNS / a.run / "model.pt"
    save(net.cpu(), path, {"run": a.run})
    from pathlib import Path
    entry = {"model": "cfc", "run": a.run, "kind": "finetune", "base": Path(a.init).parent.name if a.init else "scratch",
             "data": f"{len(tr)} training episodes (windows of {WINDOW}, {BURN_IN} warm-up steps)"
                     + (f", incl. {len(a.extra)} DAgger files" if a.extra else ""),
             "params": {"hidden": a.hidden, "steps": a.steps, "best_step": best_step, "lr": a.lr, "batch": a.batch,
                        "weights": params(net), "odometry": False},
             "train_s": round(train_s, 1), "offline": best[1], "notes": a.notes}
    print("offline", best[1], "best step", best_step, f"train {train_s:.0f} s", flush=True)
    if not a.no_bench:
        summary = benchmark.run(functools.partial(CfCDriver, str(path), "cpu"), workers=a.workers)
        entry["bench"], entry["score"] = summary, benchmark.score(summary)
        print({k: summary[k] for k in ("no_cats", "4_cats", "latency_ms")}, "score", entry["score"], flush=True)
    tracker.log(entry)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
