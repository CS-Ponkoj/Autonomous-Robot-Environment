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
from .route import ROUTE_FEATURES, make_route_source, route_spec, source_from_spec

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
        """route None (or the trained source's own name): the source exactly as the checkpoint
        records it (for "map": the same saved map and drift correction); another name: that
        source with its current defaults."""
        if route is None or route == self.spec.get("name", "online"):
            self.source = source_from_spec(self.spec)
        else:
            self.source = make_route_source(route)
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


OUTCOMES = ("timeout", "success", "collision")


class Episodes:
    """DAgger episodes in one .npz: x (N, 78), t (N,), route (N, 4), cmd (N, 2), lens (episodes,),
    outcome (episodes,: an index into OUTCOMES). Loaded episodes carry a key (file name and index)
    that fixes their train or validation side for good (split_episodes)."""

    @staticmethod
    def save(path: Path, eps: list[dict]) -> None:
        from ...common.features import FEATURES
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = lambda k, n: np.concatenate([np.asarray(e[k], np.float32).reshape(-1, n) for e in eps])  # noqa: E731
        np.savez_compressed(path, x=rows("x", FEATURES), t=rows("t", 1)[:, 0], route=rows("route", 4), cmd=rows("cmd", 2),
                            lens=np.array([len(np.asarray(e["x"]).reshape(-1, FEATURES)) for e in eps]),
                            outcome=np.array([OUTCOMES.index(e.get("outcome", "timeout")) for e in eps]))

    @staticmethod
    def load(paths, allow_inconsistent: bool = False) -> list[dict]:
        """Episodes of these files; a file whose .json is marked label_inconsistent (its expert
        planned another route than the policy read) is refused unless allow_inconsistent (analysis)."""
        import json
        out = []
        for f in paths:
            meta = Path(f).with_suffix(".json")
            mark = json.loads(meta.read_text(encoding="utf-8")).get("label_inconsistent") if meta.exists() else None
            if mark and not allow_inconsistent:
                raise ValueError(f"{f} is marked label_inconsistent: {mark}")
            d = np.load(f)
            cut = np.cumsum(d["lens"])[:-1]
            outcomes = d["outcome"] if "outcome" in d.files else None
            for i, (x, t, r, c) in enumerate(zip(*(np.split(d[k], cut) for k in ("x", "t", "route", "cmd")))):
                if len(x):
                    out.append({"x": x.astype(np.float32), "t": t.astype(np.float32),
                                "route": r.astype(np.float32), "cmd": c.astype(np.float32),
                                "outcome": OUTCOMES[int(outcomes[i])] if outcomes is not None else None,
                                "key": f"{Path(f).name}:{i}"})
        return out


def split_episodes(eps: list[dict], share: float = 0.1) -> tuple[list, list]:
    """Hold out whole episodes (about `share`) for model selection. The side of each episode is
    fixed by a checksum of its key (file and index), so an episode validated in one round is never
    trained on in a later one. Needs at least 2 episodes; at least one lands on each side."""
    import zlib
    if len(eps) < 2:
        raise ValueError(f"route DAgger needs at least 2 episodes to hold one out (got {len(eps)})")
    held = [zlib.crc32(e.get("key", str(i)).encode()) % 1000 < share * 1000 for i, e in enumerate(eps)]
    if not any(held):  # small runs: the episode with the smallest checksum
        held[min(range(len(eps)), key=lambda i: zlib.crc32(eps[i].get("key", str(i)).encode()))] = True
    if all(held):
        held[0] = False
    return [e for e, h in zip(eps, held) if not h], [e for e, h in zip(eps, held) if h]


def flat(eps: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """All steps as (N, 82) inputs and (N, 2) expert commands."""
    x = np.concatenate([np.concatenate([e["x"], e["route"]], 1) for e in eps]).astype(np.float32)
    return x, np.concatenate([e["cmd"] for e in eps]).astype(np.float32)
