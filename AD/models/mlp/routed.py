"""Route-input adapter for the MLP family (the interface of AD/models/hybrid/routed.py): the
continuous-head MLP with the route source's 4 values after the 78 features (its first layer
widened with zero weights, so before training it drives exactly as its base), trained by DAgger
(AD/models/hybrid/route_dagger.py). The MLP's own files are unchanged."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from robot_env.types import Command, Decision, Observation

from ...common import dataset
from ...common.features import FEATURES, ODOMETRY, labels_of
from ..hybrid.route import ROUTE_FEATURES
from ..hybrid.routekit import RouteDriver, flat, spec_of, widen_linear
from .model import SCALE, Net

FAMILY = "mlp"


def params(net: Net) -> int:
    return sum(p.numel() for p in net.parameters())


def _net(ck: dict) -> Net:
    net = Net(**ck["arch"])
    n_in = ck.get("n_in", FEATURES)
    if n_in != FEATURES:
        net.body[0] = nn.Linear(n_in, net.body[0].out_features)
    net.load_state_dict(ck["state"])
    net.arch, net.n_in, net.no_odometry = ck["arch"], n_in, ck.get("no_odometry", False)
    return net.eval()


def load(path) -> Net:
    return _net(torch.load(path, map_location="cpu", weights_only=False))


def save(net: Net, path: Path, spec: dict, extra: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"arch": net.arch, "n_in": net.n_in, "no_odometry": net.no_odometry,
                "state": {k: v.cpu() for k, v in net.state_dict().items()}, "route_spec": spec, **(extra or {})}, path)


def widen_from(base_path) -> Net:
    ck = torch.load(base_path, map_location="cpu", weights_only=False)
    assert ck["arch"].get("head") == "cont", "route DAgger needs a continuous-head MLP (e.g. mlp_cont)"
    assert ck.get("n_in", FEATURES) == FEATURES, "already a route MLP"
    net = _net(ck)
    net.body[0] = widen_linear(net.body[0], at=FEATURES)
    net.n_in = FEATURES + ROUTE_FEATURES
    return net.eval()


class Driver(RouteDriver):
    """A route MLP as a robot_env driver (CPU by default)."""

    def __init__(self, path, device: str | None = "cpu", route: str | None = None):
        torch.set_num_threads(1)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        ck = torch.load(path, map_location="cpu", weights_only=False)
        self.net = _net(ck).to(self.device)
        assert self.net.n_in == FEATURES + ROUTE_FEATURES, "not a route MLP checkpoint"
        self.spec = spec_of(ck)
        self._init_route(route)
        self.name = f"mlp:{Path(path).parent.name}"

    def decide(self, obs: Observation) -> Decision:
        x = self.route_input(obs)
        if self.net.no_odometry:
            x[ODOMETRY] = 0.0
        with torch.no_grad():
            v, w = self.net(torch.from_numpy(x)[None].to(self.device))[0].cpu().tolist()
        return Decision(Command(v, w), obs.seq)


def fit(net: Net, train: list[dict], valid: list[dict], a) -> tuple[Net, dict]:
    """The MLP's continuous loss (each axis as a share of its cap) on the DAgger states; the epoch
    with the lowest held-out error kept. a: epochs, lr, device."""
    xt, yt = flat(train)
    xv, yv = flat(valid)
    if net.no_odometry:
        xt[:, ODOMETRY] = 0.0
        xv[:, ODOMETRY] = 0.0
    dev = a.device
    net = net.to(dev)
    scale = SCALE.to(dev)
    X, Y = torch.from_numpy(xt).to(dev), torch.from_numpy(yt).to(dev)
    XV = torch.from_numpy(xv).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    best, best_state = None, None
    for ep in range(a.epochs):
        net.train()
        perm = torch.randperm(len(X), device=dev)
        for i in range(0, len(X), 1024):
            idx = perm[i:i + 1024]
            loss = nn.functional.mse_loss(net(X[idx]) / scale, Y[idx] / scale)
            opt.zero_grad()
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            pv = net(XV).cpu().numpy()
        r = dataset.accuracy(np.asarray(labels_of(pv)), np.asarray(labels_of(yv)))
        err = np.abs(pv - yv)
        r["mae_v"], r["mae_omega"] = round(float(err[:, 0].mean()), 4), round(float(err[:, 1].mean()), 4)
        key = -(r["mae_v"] / float(SCALE[0]) + r["mae_omega"] / float(SCALE[1]))
        if best is None or key > best[0]:
            best = (key, {**r, "best_epoch": ep + 1})
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        print(f"epoch {ep + 1}/{a.epochs} loss {loss.item():.4f} valid acc {r['accuracy']:.3f}", flush=True)
    net.load_state_dict(best_state)
    return net.cpu().eval(), best[1]


__all__ = ["FAMILY", "Driver", "widen_from", "load", "save", "fit", "params"]
