"""Route-input adapter for the CfC family (the interface of AD/models/hybrid/routed.py): the CfC
with the route source's 4 values next to the goal values in its scalar branch, its memory carried
through each episode, trained by DAgger on whole episodes (AD/models/hybrid/route_dagger.py)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from robot_env.types import Command, Decision, Observation

from ...common.features import labels_of
from ..hybrid.route import ROUTE_FEATURES
from ..hybrid.routekit import RouteDriver, spec_of, widen_linear
from .model import SCALARS, CfCPolicy, load as load_cfc, params, save as save_cfc

FAMILY = "cfc"
load = load_cfc


def widen_from(base_path) -> CfCPolicy:
    base = load_cfc(base_path)
    assert base.extra == 0, "already a route CfC"
    net = CfCPolicy(base.hidden_size, base.width, extra=ROUTE_FEATURES)
    state = base.state_dict()
    state["scalars.0.weight"] = widen_linear(base.scalars[0], at=SCALARS).weight.detach()
    mask = torch.ones(len(state["mask"]) + ROUTE_FEATURES)
    mask[:len(state["mask"])] = state["mask"]
    state["mask"] = mask
    net.load_state_dict(state)
    return net.eval()


def save(net: CfCPolicy, path: Path, spec: dict, extra: dict | None = None) -> None:
    save_cfc(net.cpu(), path, {"route_spec": spec, **(extra or {})})


class Driver(RouteDriver):
    """A route CfC as a robot_env driver: route source and memory both carried through the episode."""

    def __init__(self, path, device: str | None = "cpu", route: str | None = None):
        torch.set_num_threads(1)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.net = load_cfc(path).to(self.device)
        assert self.net.extra == ROUTE_FEATURES, "not a route CfC checkpoint"
        self.spec = spec_of(torch.load(path, map_location="cpu", weights_only=False))
        self._init_route(route)
        self.name = f"cfc:{Path(path).parent.name}"
        self.h = self.net.initial(1, self.device)

    def reset(self, initial_pose=None, **kw) -> None:
        super().reset(initial_pose=initial_pose, **kw)
        self.h = self.net.initial(1, self.device)

    def decide(self, obs: Observation) -> Decision:
        x = self.route_input(obs)
        with torch.no_grad():
            out, self.h = self.net.step(torch.from_numpy(x)[None].to(self.device), self.h)
        v, w = out[0].cpu().tolist()
        return Decision(Command(v, w), obs.seq)


def _episodes(eps: list[dict]) -> list[dict]:
    """In the CfC trainer's form: x (T, 82), cmd (T, 2), action (T,)."""
    return [{"x": np.concatenate([e["x"], e["route"]], 1).astype(np.float32), "cmd": e["cmd"],
             "action": np.asarray(labels_of(e["cmd"]))} for e in eps]


def fit(net: CfCPolicy, train: list[dict], valid: list[dict], a) -> tuple[CfCPolicy, dict]:
    """Windows of whole-episode sequences (the CfC trainer's windows and evaluation), the epoch
    with the lowest held-out (v, omega) error kept. a: steps, batch, lr, device."""
    from .train import evaluate, windows
    from .model import V_CAP, W_CAP
    tr, va = _episodes(train), _episodes(valid)
    rng = np.random.default_rng(0)
    net = net.to(a.device)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)
    scale = torch.tensor([V_CAP, W_CAP], device=a.device)
    best, best_state = None, None
    for step in range(1, a.steps + 1):
        net.train()
        X, Y, M = (torch.from_numpy(v).to(a.device) for v in windows(tr, a.batch, rng))
        out, _ = net(X)
        loss = ((((out - Y) / scale) ** 2).sum(-1) * M).sum() / M.sum()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % max(1, a.steps // 5) == 0 or step == a.steps:
            r = evaluate(net, va, a.device)
            key = -(r["mae_v"] / V_CAP + r["mae_omega"] / W_CAP)
            if best is None or key > best[0]:
                best = (key, {**r, "best_step": step})
                best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            print(f"step {step} loss {loss.item():.4f} valid acc {r['accuracy']:.3f}", flush=True)
    net.load_state_dict(best_state)
    return net.cpu().eval(), best[1]


__all__ = ["FAMILY", "Driver", "widen_from", "load", "save", "fit", "params"]
