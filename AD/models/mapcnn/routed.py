"""Route-input adapter for the map CNN family (the interface of AD/models/hybrid/routed.py): the
map CNN with the route source's 4 values next to the goal values in its scalar branch (widened
with zero weights, so before training it drives exactly as its base), trained by DAgger
(AD/models/hybrid/route_dagger.py). Its local maps are rebuilt from each DAgger episode's
features and times with maps.replay, as for the recordings. The map CNN's own files are unchanged."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from robot_env.types import Command, Decision, Observation

from ...common.features import FEATURES, labels_of
from ..hybrid.route import ROUTE_FEATURES
from ..hybrid.routekit import RouteDriver, spec_of, widen_linear
from . import maps
from .policy import SCALARS, MapPolicy, load as load_map, params

FAMILY = "mapcnn"


class RouteMapPolicy(MapPolicy):
    """MapPolicy whose scalar branch also reads `extra` route values after the 78 features."""

    def __init__(self, width: int = 32, extra: int = ROUTE_FEATURES):
        super().__init__(width)
        self.extra = extra
        self.scalars[0] = nn.Linear(SCALARS + extra, self.scalars[0].out_features)
        mask = torch.ones(FEATURES + extra)
        mask[:FEATURES] = self.mask
        self.mask = mask  # replaces the registered buffer's value (a buffer stays a buffer)


def load(path) -> RouteMapPolicy:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    net = RouteMapPolicy(ck.get("width", 32), ck.get("extra", ROUTE_FEATURES))
    net.load_state_dict(ck["state"])
    return net.eval()


def save(net: RouteMapPolicy, path: Path, spec: dict, extra: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"width": net.width, "extra": net.extra, "state": net.cpu().state_dict(), "route_spec": spec,
                **(extra or {})}, path)


def widen_from(base_path) -> RouteMapPolicy:
    base = load_map(base_path)
    net = RouteMapPolicy(base.width)
    state = base.state_dict()
    state["scalars.0.weight"] = widen_linear(base.scalars[0], at=SCALARS).weight.detach()
    mask = torch.ones(FEATURES + ROUTE_FEATURES)
    mask[:FEATURES] = state["mask"]
    state["mask"] = mask
    net.load_state_dict(state)
    return net.eval()


class Driver(RouteDriver):
    """A route map CNN as a robot_env driver: its local map and the route source both built as it
    goes (CPU by default)."""

    def __init__(self, path, device: str | None = "cpu", route: str | None = None):
        torch.set_num_threads(1)
        self.device = device or "cpu"
        self.net = load(path).to(self.device)
        self.spec = spec_of(torch.load(path, map_location="cpu", weights_only=False))
        self._init_route(route)
        self.map = maps.LocalMap()
        self.name = f"mapcnn:{Path(path).parent.name}"

    def reset(self, initial_pose=None, **kw) -> None:
        super().reset(initial_pose=initial_pose, **kw)
        self.map.reset()

    def decide(self, obs: Observation) -> Decision:
        x = self.route_input(obs)
        m = self.map.step(self.last_x, obs.time)  # the map from the 78 features, as in training
        with torch.no_grad():
            v, w = self.net(torch.from_numpy(x)[None].to(self.device), torch.from_numpy(m)[None].to(self.device))[0].cpu().tolist()
        return Decision(Command(float(v), float(w)), obs.seq)


def _dataset(eps: list[dict]) -> dict:
    """The map CNN trainer's rows: x (N, 82), maps (N, 3, 32, 32) replayed per episode, cmd, action."""
    x = np.concatenate([np.concatenate([e["x"], e["route"]], 1) for e in eps]).astype(np.float32)
    m = np.concatenate([maps.replay(e["x"], e["t"]) for e in eps])
    cmd = np.concatenate([e["cmd"] for e in eps]).astype(np.float32)
    return {"x": x, "maps": m, "cmd": cmd, "action": np.asarray(labels_of(cmd))}


def fit(net: RouteMapPolicy, train: list[dict], valid: list[dict], a) -> tuple[RouteMapPolicy, dict]:
    from .train import fit as fit_map
    offline, best_ep = fit_map(net, _dataset(train), _dataset(valid), a.epochs, 1024, a.lr, a.device)
    return net.cpu().eval(), {**offline, "best_epoch": best_ep + 1}


__all__ = ["FAMILY", "Driver", "widen_from", "load", "save", "fit", "params"]
