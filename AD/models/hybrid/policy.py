"""The hybrid policy: the 1D-CNN with the route sensor's 4 values (route.py) as extra inputs to its
scalar branch, and its driver.

widen() turns a trained continuous CNN (dagger_cont_r3) into a hybrid with the new inputs' weights
at zero, so before training it drives exactly as that CNN; DAgger then teaches it to use them."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from robot_env.types import Command, Decision, Observation

from ..cnn.policy import SCALARS, CNNPolicy, load as load_cnn
from .route import ROUTE_FEATURES
from .routekit import RouteDriver, spec_of


def widen(base: CNNPolicy) -> CNNPolicy:
    """The same network with ROUTE_FEATURES more inputs to the scalar branch, weighted zero."""
    assert base.frames == 1 and base.extra == 0
    net = CNNPolicy(base.head, 1, base.width, base.odometry, extra=ROUTE_FEATURES)
    state = base.state_dict()
    for k, v in net.state_dict().items():
        if k == "scalars.0.weight":  # (64, SCALARS) -> (64, SCALARS + 4): the old columns, then zeros
            w = torch.zeros_like(v)
            w[:, :SCALARS] = state[k]
            state[k] = w
        elif k == "mask":
            m = torch.ones_like(v)
            m[:len(state[k])] = state[k]
            state[k] = m
    net.load_state_dict(state)
    return net.eval()


class HybridDriver(RouteDriver):
    """A hybrid CNN as a robot_env driver: the route source reads every observation, the network
    decides. route: run another source than the checkpoint's (e.g. "map"). `last_x` is the
    82-value input of the latest decision and `last_route` its 4 route values (for DAgger)."""

    def __init__(self, path, device: str | None = "cpu", route: str | None = None):
        torch.set_num_threads(1)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.net = load_cnn(path).to(self.device)
        assert self.net.extra == ROUTE_FEATURES, "not a hybrid checkpoint"
        self.spec = spec_of(torch.load(path, map_location="cpu", weights_only=False))
        self._init_route(route)
        self.name = f"hybrid:{Path(path).parent.name}"

    def decide(self, obs: Observation) -> Decision:
        x = self.route_input(obs)
        self.last_x = x  # all 82 values (what AD/models/cnn/dagger.py stores for kind "hybrid")
        with torch.no_grad():
            v, w = self.net(torch.from_numpy(x)[None].to(self.device))[0].cpu().tolist()
        return Decision(Command(v, w), obs.seq)
