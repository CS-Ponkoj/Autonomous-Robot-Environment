"""A Decision Transformer (Hugging Face transformers.DecisionTransformerModel) trained on the
built-in driver's recorded episodes, then benchmarked as a driver.

    .venv\\Scripts\\python -m AD.models.dt.train --run dt_k20

States are AD.common.features.vector(obs) (78 numbers). Actions are the driver's continuous
(v, omega), each divided by its cap (so in [-1, 1]). The reward of a step is its progress toward
the goal (the drop in goal distance, in metres) plus GOAL_BONUS on the step that arrives; the
return-to-go is what is still to come. Driving, the model is asked for a target return (the start
distance plus the bonus) and that is reduced by the progress made, as in the original paper.
Checkpoints: AD/dt/runs/<run>/."""

from __future__ import annotations

import argparse
import functools
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

from robot_env import config as C
from robot_env.types import Command, Decision, Observation

from ...common import benchmark, dataset, tracker
from ...common.features import FEATURES, LABEL_LEVEL, LIDAR_BINS, labels_of, vector

RUNS = tracker.AD / "models" / "dt" / "runs"
CAPS = np.array(C.SPEED_LEVELS[LABEL_LEVEL], np.float32)  # 0.30 m/s, 1.0 rad/s
GOAL_BONUS = 2.0  # return units (metres of progress) for arriving
RTG_SCALE = 10.0  # returns are divided by this inside the model
MAX_EP_LEN = int(C.EPISODE_TIME_LIMIT / C.DECISION_PERIOD) + 2
DIST = LIDAR_BINS  # index of the goal distance (divided by 10) in the feature vector


def episodes(seeds) -> list[dict]:
    """The recorded episodes: states, scaled actions, rewards, returns-to-go (per seed)."""
    d = dataset.load(seeds)
    out = []
    for s in np.unique(d["seed"]):
        sel = d["seed"] == s
        x, cmd = d["x"][sel], d["cmd"][sel]
        dist = x[:, DIST] * 10.0  # metres (capped at 10, as the feature is)
        r = np.zeros(len(x), np.float32)
        r[:-1] = dist[:-1] - dist[1:]
        from ..cnn.train import outcome
        arrived = outcome(int(s)) == "success"
        r[-1] = GOAL_BONUS if arrived else 0.0
        rtg = np.cumsum(r[::-1])[::-1].astype(np.float32)
        out.append({"seed": int(s), "x": x, "a": cmd / CAPS, "rtg": rtg, "action": d["action"][sel]})
    return out


def make_model(hidden: int, layers: int, heads: int, k: int):
    from transformers import DecisionTransformerConfig, DecisionTransformerModel
    cfg = DecisionTransformerConfig(state_dim=FEATURES, act_dim=2, hidden_size=hidden, n_layer=layers, n_head=heads,
                                    n_inner=4 * hidden, max_ep_len=MAX_EP_LEN, n_positions=3 * k + 3,
                                    action_tanh=True, resid_pdrop=0.1, attn_pdrop=0.1, embd_pdrop=0.1)
    return DecisionTransformerModel(cfg)


def batch(eps: list[dict], k: int, n: int, rng: np.random.Generator, device: str):
    """n windows of k steps (left-padded at an episode's start), chosen in proportion to length."""
    lens = np.array([len(e["x"]) for e in eps])
    pick = rng.choice(len(eps), n, p=lens / lens.sum())
    S = np.zeros((n, k, FEATURES), np.float32)
    A = np.zeros((n, k, 2), np.float32)
    R = np.zeros((n, k, 1), np.float32)
    T = np.zeros((n, k), np.int64)
    M = np.zeros((n, k), np.float32)
    for i, j in enumerate(pick):
        e = eps[j]
        end = int(rng.integers(1, len(e["x"]) + 1))
        lo = max(0, end - k)
        m = end - lo
        S[i, -m:], A[i, -m:], R[i, -m:, 0] = e["x"][lo:end], e["a"][lo:end], e["rtg"][lo:end] / RTG_SCALE
        T[i, -m:], M[i, -m:] = np.arange(lo, end), 1.0
    t = lambda v: torch.from_numpy(v).to(device)  # noqa: E731
    return t(S), t(A), t(R), t(T), t(M)


def predict(model, S, A, R, T, M, actions_in: bool = True):
    """Action predictions; actions_in=False hides the past actions (all zero), so the model
    cannot simply repeat what it did (the copycat shortcut) and must read the states."""
    if not actions_in:
        A = torch.zeros_like(A)
    out = model(states=S, actions=A, rewards=torch.zeros_like(R), returns_to_go=R, timesteps=T,
                attention_mask=M, return_dict=True)
    return out.action_preds


def evaluate(model, eps: list[dict], k: int, device: str, actions_in: bool = True) -> dict:
    """Teacher-forced: the action predicted at every step of the validation episodes."""
    model.eval()
    preds, true_a, labels = [], [], []
    with torch.no_grad():
        for e in eps:
            n = len(e["x"])
            for start in range(0, n, 256):
                idx = np.arange(start, min(n, start + 256))
                S = np.zeros((len(idx), k, FEATURES), np.float32)
                A = np.zeros((len(idx), k, 2), np.float32)
                R = np.zeros((len(idx), k, 1), np.float32)
                T = np.zeros((len(idx), k), np.int64)
                M = np.zeros((len(idx), k), np.float32)
                for i, end in enumerate(idx + 1):
                    lo = max(0, end - k)
                    m = end - lo
                    S[i, -m:], A[i, -m:], R[i, -m:, 0] = e["x"][lo:end], e["a"][lo:end], e["rtg"][lo:end] / RTG_SCALE
                    A[i, -1] = 0.0  # the action being predicted is not given
                    T[i, -m:], M[i, -m:] = np.arange(lo, end), 1.0
                t = lambda v: torch.from_numpy(v).to(device)  # noqa: E731
                preds.append(predict(model, t(S), t(A), t(R), t(T), t(M), actions_in)[:, -1].cpu().numpy())
            true_a.append(e["a"])
            labels.append(e["action"])
    p, a = np.concatenate(preds) * CAPS, np.concatenate(true_a) * CAPS
    lab = np.asarray(labels_of(p))
    r = dataset.accuracy(lab, np.concatenate(labels))
    r["mae_v"], r["mae_omega"] = round(float(np.abs(p - a)[:, 0].mean()), 4), round(float(np.abs(p - a)[:, 1].mean()), 4)
    return r


class DTDriver:
    """A trained Decision Transformer as a robot_env driver: on the GPU when there is one (the
    simulation stays on the CPU), else on one CPU thread."""

    def __init__(self, path, device: str | None = None):
        """device None: the GPU if there is one; "cpu" for many side-by-side benchmark workers
        (each saves the GPU context's host memory)."""
        torch.set_num_threads(1)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        ck = torch.load(path, map_location="cpu", weights_only=False)
        self.k, self.actions_in = ck["k"], ck.get("actions_in", True)
        self.model = make_model(ck["hidden"], ck["layers"], ck["heads"], self.k)
        self.model.load_state_dict(ck["state"])
        self.model.to(self.device).eval()
        self.name = f"dt:{Path(path).parent.name}"
        self.reset()

    def reset(self) -> None:
        self.S, self.A, self.R = deque(maxlen=self.k), deque(maxlen=self.k), deque(maxlen=self.k)
        self.t, self.rtg, self.last = 0, None, None

    def decide(self, obs: Observation) -> Decision:
        x = vector(obs)
        dist = float(x[DIST]) * 10.0
        if self.rtg is None:
            self.rtg = dist + GOAL_BONUS  # the target: all the way, and arrive
        else:
            self.rtg -= self.last - dist  # what has been gained is no longer to come
        self.last = dist
        self.S.append(x)
        self.R.append(self.rtg / RTG_SCALE)
        self.A.append(np.zeros(2, np.float32))  # the action to predict
        m = len(self.S)
        S = np.zeros((1, self.k, FEATURES), np.float32)
        A = np.zeros((1, self.k, 2), np.float32)
        R = np.zeros((1, self.k, 1), np.float32)
        T = np.zeros((1, self.k), np.int64)
        M = np.zeros((1, self.k), np.float32)
        S[0, -m:], A[0, -m:], R[0, -m:, 0] = np.array(self.S), np.array(self.A), np.array(self.R)
        T[0, -m:], M[0, -m:] = np.arange(self.t - m + 1, self.t + 1).clip(0, MAX_EP_LEN - 1), 1.0
        with torch.no_grad():
            a = predict(self.model, *(torch.from_numpy(v).to(self.device) for v in (S, A, R, T, M)),
                        self.actions_in)[0, -1].cpu().numpy()
        self.A[-1] = a.astype(np.float32)  # what it did, for the next step's context
        self.t += 1
        v, w = (a * CAPS).tolist()
        return Decision(Command(v, w), obs.seq)


def bench_saved(a, tr: list[dict], va: list[dict], device: str) -> int:
    """Evaluate, benchmark and log a run whose training finished but whose benchmark did not."""
    path = RUNS / a.run / "model.pt"
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = make_model(ck["hidden"], ck["layers"], ck["heads"], ck["k"])
    model.load_state_dict(ck["state"])
    offline = evaluate(model.to(device), va, ck["k"], device, ck.get("actions_in", True))
    print("offline", offline, flush=True)
    summary = benchmark.run(functools.partial(DTDriver, str(path), device=a.bench_device), workers=a.workers)
    entry = {"model": "dt", "run": a.run, "kind": "finetune", "base": "scratch",
             "data": f"{len(tr)} training episodes, {sum(len(e['x']) for e in tr)} steps",
             "params": {"k": ck["k"], "actions_in": ck.get("actions_in", True), "hidden": ck["hidden"],
                        "layers": ck["layers"], "heads": ck["heads"], "steps": a.steps, "lr": a.lr, "batch": a.batch,
                        "weights": sum(v.numel() for v in ck["state"].values()), "goal_bonus": GOAL_BONUS},
             "offline": offline, "bench": summary, "score": benchmark.score(summary), "notes": a.notes}
    print({kk: summary[kk] for kk in ("no_cats", "4_cats", "latency_ms")}, "score", entry["score"], flush=True)
    tracker.log(entry)
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--run", required=True)
    p.add_argument("--k", type=int, default=20, help="context length (steps)")
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--train", default=None, help="seed range a-b (default: dataset.TRAIN)")
    p.add_argument("--valid", default=None, help="seed range a-b (default: dataset.VALID)")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--no-bench", action="store_true")
    p.add_argument("--no-actions", action="store_true", help="hide the past actions from the model (against the copycat shortcut)")
    p.add_argument("--bench-device", default=None, choices=("cpu", "cuda"), help="where benchmark workers run the model")
    p.add_argument("--bench-only", action="store_true",
                   help="no training: evaluate, benchmark and log the run's saved checkpoint (an interrupted run)")
    p.add_argument("--lock", default=None, help="hold the machine lock as this owner while training (not while benchmarking)")
    p.add_argument("--notes", default="")
    a = p.parse_args(argv)
    rng_of = lambda s, d: d if s is None else range(int(s.split("-")[0]), int(s.split("-")[1]) + 1)  # noqa: E731
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tr, va = episodes(rng_of(a.train, dataset.TRAIN)), episodes(rng_of(a.valid, dataset.VALID))
    if a.bench_only:
        return bench_saved(a, tr, va, device)
    model = make_model(a.hidden, a.layers, a.heads, a.k).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 1000) * max(0.05, 1 - s / a.steps))
    from ..cnn.train import machine_lock
    machine_lock("acquire", a.lock)
    t0, best, best_state, best_step = time.time(), None, None, 0
    for step in range(1, a.steps + 1):
        model.train()
        S, A, R, T, M = batch(tr, a.k, a.batch, rng, device)
        pred = predict(model, S, A, R, T, M, not a.no_actions)
        loss = (((pred - A) ** 2).sum(-1) * M).sum() / M.sum()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
        opt.step()
        sched.step()
        if step % 2000 == 0 or step == a.steps:
            r = evaluate(model, va, a.k, device, not a.no_actions)
            key = -(r["mae_v"] / CAPS[0] + r["mae_omega"] / CAPS[1])
            if best is None or key > best[0]:
                best, best_step = (key, r), step
                best_state = {kk: v.detach().cpu().clone() for kk, v in model.state_dict().items()}
            print(f"step {step} loss {loss.item():.4f} valid acc {r['accuracy']:.3f} mae v {r['mae_v']:.3f} w {r['mae_omega']:.3f}",
                  flush=True)
    train_s = time.time() - t0
    machine_lock("release", a.lock)
    path = RUNS / a.run / "model.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"k": a.k, "actions_in": not a.no_actions, "hidden": a.hidden, "layers": a.layers, "heads": a.heads, "state": best_state}, path)
    n_params = sum(v.numel() for v in best_state.values())
    entry = {"model": "dt", "run": a.run, "kind": "finetune", "base": "scratch",
             "data": f"{len(tr)} training episodes, {sum(len(e['x']) for e in tr)} steps",
             "params": {"k": a.k, "actions_in": not a.no_actions, "hidden": a.hidden, "layers": a.layers, "heads": a.heads, "steps": a.steps,
                        "best_step": best_step, "lr": a.lr, "batch": a.batch, "weights": n_params,
                        "goal_bonus": GOAL_BONUS},
             "train_s": round(train_s, 1), "offline": best[1], "notes": a.notes}
    print("offline", best[1], "best step", best_step, f"train {train_s:.0f} s", flush=True)
    if not a.no_bench:
        summary = benchmark.run(functools.partial(DTDriver, str(path), device=a.bench_device), workers=a.workers)
        entry["bench"], entry["score"] = summary, benchmark.score(summary)
        print({kk: summary[kk] for kk in ("no_cats", "4_cats", "latency_ms")}, "score", entry["score"], flush=True)
    tracker.log(entry)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
