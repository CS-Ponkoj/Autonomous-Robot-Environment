"""Route-input adapter for the CNN family (the hybrid): the interface route_dagger.py uses for
every family.

    FAMILY                      the model folder the runs and log entries go to
    widen_from(base_path)       the base family checkpoint with 4 zero-weight route inputs
    load(path) / save(...)      checkpoints that carry the route source's spec
    Driver(path, device, route) the driver (route source inside decide())
    fit(net, train, valid, a)   train on DAgger episodes; returns (net, offline result)
    params(net)"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ...common.features import labels_of
from ..cnn.policy import load as load_cnn, params, save as save_cnn
from .policy import HybridDriver, widen
from .routekit import flat

FAMILY = "hybrid"
Driver = HybridDriver
load = load_cnn


def widen_from(base_path) -> object:
    return widen(load_cnn(base_path))


def save(net, path: Path, spec: dict, extra: dict | None = None) -> None:
    save_cnn(net.cpu(), path, {"route_spec": spec, **(extra or {})})


def _dataset(eps: list[dict]) -> dict:
    x, cmd = flat(eps)
    return {"x": x, "cmd": cmd, "action": np.asarray(labels_of(cmd))}


def fit(net, train: list[dict], valid: list[dict], a) -> tuple[object, dict]:
    from ..cnn.train import fit as fit_cnn
    offline, best_ep = fit_cnn(net, _dataset(train), _dataset(valid), a.epochs, 1024, a.lr, a.device)
    offline["best_epoch"] = best_ep + 1
    return net, offline


__all__ = ["FAMILY", "Driver", "widen_from", "load", "save", "fit", "params"]
