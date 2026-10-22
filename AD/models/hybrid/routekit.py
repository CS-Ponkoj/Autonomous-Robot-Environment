"""Shared pieces for route-input policies of every family (cnn hybrid, cfc, dt, mlp, mapcnn).

- widen_linear: a Linear layer with ROUTE_FEATURES more inputs, inserted at a given column with
  zero weights, so a widened network computes exactly what it did before training.
- RouteDriver: the part of a driver that runs the route source: built from the checkpoint's
  route spec (or a source named at evaluation), reset with the episode's start pose, and its
  latest 4 values kept in last_route (with the 78 features in last_x) for DAgger.
- Episodes: the DAgger state store shared by all families: per episode the 78 features, the
  observation times, the 4 route values and the built-in driver's (v, omega)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from robot_env.types import Observation

from ...common.features import vector
from .route import ROUTE_FEATURES, make_route_source, route_spec

DEFAULT_SPEC = {"name": "online", "len_scale": 10.0}  # checkpoints from before route specs existed


def widen_linear(layer: nn.Linear, at: int, extra: int = ROUTE_FEATURES) -> nn.Linear:
    """layer with `extra` new input columns inserted before column `at`, weighted zero."""
    new = nn.Linear(layer.in_features + extra, layer.out_features, bias=layer.bias is not None)
    with torch.no_grad():
        w = torch.zeros(layer.out_features, layer.in_features + extra)
        w[:, :at] = layer.weight[:, :at]
        w[:, at + extra:] = layer.weight[:, at:]
        new.weight.copy_(w)
        if layer.bias is not None:
            new.bias.copy_(layer.bias)
    return new


class RouteDriver:
    """Route handling for a driver: subclasses set self.spec (from the checkpoint) and call
    _init_route(route) in __init__, then route_input(obs) in decide()."""

    def _init_route(self, route: str | None) -> None:
        name = route or self.spec.get("name", "online")
        self.source = make_route_source(name)
        self.last_x = self.last_route = None

    def reset(self, initial_pose=None, **kw) -> None:
        self.source.reset(initial_pose=initial_pose, **kw)
        self.last_x = self.last_route = None

    def route_input(self, obs: Observation) -> np.ndarray:
        """The 82 inputs: features.vector(obs) and the route source's 4 values (both kept)."""
        x = vector(obs)
        r = self.source.read(obs)
        self.last_x, self.last_route = x, r
        return np.concatenate([x, r]).astype(np.float32)


def spec_of(ck: dict) -> dict:
    return ck.get("route_spec", DEFAULT_SPEC)


def source_spec(name: str) -> dict:
    return route_spec(make_route_source(name))


class Episodes:
    """DAgger episodes in one .npz: x (N, 78), t (N,), route (N, 4), cmd (N, 2), lens (episodes,)."""

    @staticmethod
    def save(path: Path, eps: list[dict]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, x=np.concatenate([e["x"] for e in eps]), t=np.concatenate([e["t"] for e in eps]),
                            route=np.concatenate([e["route"] for e in eps]), cmd=np.concatenate([e["cmd"] for e in eps]),
                            lens=np.array([len(e["x"]) for e in eps]))

    @staticmethod
    def load(paths) -> list[dict]:
        out = []
        for f in paths:
            d = np.load(f)
            cut = np.cumsum(d["lens"])[:-1]
            for x, t, r, c in zip(*(np.split(d[k], cut) for k in ("x", "t", "route", "cmd"))):
                if len(x):
                    out.append({"x": x.astype(np.float32), "t": t.astype(np.float32),
                                "route": r.astype(np.float32), "cmd": c.astype(np.float32)})
        return out


def split_episodes(eps: list[dict], rng: np.random.Generator, share: float = 0.1) -> tuple[list, list]:
    """Hold out whole episodes (about `share`) for model selection."""
    idx = rng.permutation(len(eps))
    n = max(1, int(round(len(eps) * share))) if len(eps) > 1 else 0
    return [eps[i] for i in idx[n:]], [eps[i] for i in idx[:n]]


def flat(eps: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """All steps as (N, 82) inputs and (N, 2) expert commands."""
    x = np.concatenate([np.concatenate([e["x"], e["route"]], 1) for e in eps]).astype(np.float32)
    return x, np.concatenate([e["cmd"] for e in eps]).astype(np.float32)
