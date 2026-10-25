"""Route-input adapter for the Decision Transformer family (the interface of
AD/models/hybrid/routed.py): states of 82 values (features.vector and the route source's 4), the
past actions hidden as in dt_noact, trained by DAgger on whole episodes with return-to-go from
each episode's goal progress (as AD/models/dt/train.py does for the recordings)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from robot_env import config as C
from robot_env.types import Observation

from ...common.features import FEATURES, labels_of
from ..hybrid.route import ROUTE_FEATURES
from ..hybrid.routekit import RouteDriver, spec_of
from .train import CAPS, DIST, GOAL_BONUS, RTG_SCALE, DTDriver, evaluate, batch, make_model, predict

FAMILY = "dt"


@dataclass
class DTNet:
    """The transformer and the settings its checkpoint needs."""
    model: torch.nn.Module
    k: int
    hidden: int
    layers: int
    heads: int
    actions_in: bool
    state_dim: int


def params(net: DTNet) -> int:
    return sum(p.numel() for p in net.model.parameters())


def _from_ck(ck: dict) -> DTNet:
    sd = ck.get("state_dim", FEATURES)
    model = make_model(ck["hidden"], ck["layers"], ck["heads"], ck["k"], sd)
    model.load_state_dict(ck["state"])
    return DTNet(model.eval(), ck["k"], ck["hidden"], ck["layers"], ck["heads"], ck.get("actions_in", True), sd)


def load(path) -> DTNet:
    return _from_ck(torch.load(path, map_location="cpu", weights_only=False))


def save(net: DTNet, path: Path, spec: dict, extra: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"k": net.k, "actions_in": net.actions_in, "hidden": net.hidden, "layers": net.layers,
                "heads": net.heads, "state_dim": net.state_dim, "state": net.model.cpu().state_dict(),
                "route_spec": spec, **(extra or {})}, path)


def widen_from(base_path) -> DTNet:
    """The base DT with 4 more state inputs: zero columns in embed_state (and zero rows in the
    unused state prediction), so before training it predicts exactly as the base."""
    ck = torch.load(base_path, map_location="cpu", weights_only=False)
    assert ck.get("state_dim", FEATURES) == FEATURES, "already a route DT"
    state = dict(ck["state"])
    w = state["embed_state.weight"]
    state["embed_state.weight"] = torch.cat([w, torch.zeros(w.shape[0], ROUTE_FEATURES)], 1)
    for key in ("predict_state.0.weight", "predict_state.weight"):
        if key in state:
            v = state[key]
            state[key] = torch.cat([v, torch.zeros(ROUTE_FEATURES, v.shape[1])], 0)
    for key in ("predict_state.0.bias", "predict_state.bias"):
        if key in state:
            state[key] = torch.cat([state[key], torch.zeros(ROUTE_FEATURES)])
    return _from_ck({**ck, "state": state, "state_dim": FEATURES + ROUTE_FEATURES})


class Driver(DTDriver, RouteDriver):
    """A route DT as a robot_env driver (CPU by default)."""

    def __init__(self, path, device: str | None = "cpu", route: str | None = None):
        DTDriver.__init__(self, path, device)
        ck = torch.load(path, map_location="cpu", weights_only=False)
        assert ck.get("state_dim", FEATURES) == FEATURES + ROUTE_FEATURES, "not a route DT checkpoint"
        self.spec = spec_of(ck)
        self._init_route(route)

    def reset(self, initial_pose=None, **kw) -> None:
        self._clear()
        RouteDriver.reset(self, initial_pose=initial_pose, **kw)

    def _state(self, obs: Observation) -> np.ndarray:
        return self.route_input(obs)


def _episodes(eps: list[dict]) -> list[dict]:
    """In the DT trainer's form: x (T, 82), a (scaled commands), rtg (goal progress to come)."""
    out = []
    for e in eps:
        x = np.concatenate([e["x"], e["route"]], 1).astype(np.float32)
        dist = e["x"][:, DIST] * 10.0
        r = np.zeros(len(x), np.float32)
        r[:-1] = dist[:-1] - dist[1:]
        if e.get("outcome") is not None:  # the rollout's own outcome (Episodes stores it)
            arrived = e["outcome"] == "success"
        else:  # episode files from before outcomes were stored: rollouts stop on arrival
            arrived = dist[-1] <= C.GOAL_RADIUS + 0.15
        r[-1] = GOAL_BONUS if arrived else 0.0
        out.append({"x": x, "a": (e["cmd"] / CAPS).astype(np.float32),
                    "rtg": np.cumsum(r[::-1])[::-1].astype(np.float32), "action": np.asarray(labels_of(e["cmd"]))})
    return out


def fit(net: DTNet, train: list[dict], valid: list[dict], a) -> tuple[DTNet, dict]:
    """The DT trainer's windows and evaluation on the DAgger episodes; the best held-out step kept.
    a: steps, batch, lr, device."""
    tr, va = _episodes(train), _episodes(valid)
    rng = np.random.default_rng(0)
    model = net.model.to(a.device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 200) * max(0.05, 1 - s / a.steps))
    best, best_state = None, None
    for step in range(1, a.steps + 1):
        model.train()
        S, A, R, T, M = batch(tr, net.k, a.batch, rng, a.device)
        pred = predict(model, S, A, R, T, M, net.actions_in)
        loss = (((pred - A) ** 2).sum(-1) * M).sum() / M.sum()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
        opt.step()
        sched.step()
        if step % max(1, a.steps // 5) == 0 or step == a.steps:
            r = evaluate(model, va, net.k, a.device, net.actions_in)
            key = -(r["mae_v"] / CAPS[0] + r["mae_omega"] / CAPS[1])
            if best is None or key > best[0]:
                best = (key, {**r, "best_step": step})
                best_state = {kk: v.detach().cpu().clone() for kk, v in model.state_dict().items()}
            print(f"step {step} loss {loss.item():.4f} valid acc {r['accuracy']:.3f}", flush=True)
    model.load_state_dict(best_state)
    net.model = model.cpu().eval()
    return net, best[1]


__all__ = ["FAMILY", "Driver", "widen_from", "load", "save", "fit", "params", "RTG_SCALE"]
