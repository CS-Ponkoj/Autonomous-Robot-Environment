"""Which lights the renderer draws. The classic renderer draws at most eight lights (the headlight
one of them), the first ones in model order, so a world with more shows only its first few
everywhere. Before each render, the lights that matter to the viewpoint are switched on and the
rest off: directional lights always, then the lights in the viewpoint's room, then the nearest
others. Visual only: physics, sensors, and contacts never read lights."""

from __future__ import annotations

import math

import mujoco
import numpy as np

from .layout import region_of

GL_LIGHTS = 8  # the most the classic renderer draws, the headlight included
FADE = 0.3  # s for a light to fade out, or in, when the window's choice changes (no popping)
# the top view shows the whole floor: a fixed set, one light per room
TOP_LIGHTS = ("light_office", "light_lab", "light_storage", "light_reception", "light_corridor_w2")


class Lights:
    def __init__(self, model: mujoco.MjModel):
        self.model = model
        self.base = model.light_diffuse.copy()
        self.shadowing = model.light_castshadow.copy() > 0  # the lights the world lets cast shadows
        self.directional = model.light_type == mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
        # the room each light shines into: a little way along its beam from where it hangs (a
        # window's light sits in the wall)
        self.region = []
        for k in range(model.nlight):
            x, y = model.light_pos[k, :2]
            dx, dy = model.light_dir[k, :2]
            n = math.hypot(dx, dy)
            if n > 1e-6:
                x, y = x + 0.3 * dx / n, y + 0.3 * dy / n
            self.region.append(region_of(float(x), float(y)))
        names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_LIGHT, k) for k in range(model.nlight)]
        top = [k for k, name in enumerate(names) if name in TOP_LIGHTS]
        self.top = list(np.flatnonzero(self.directional)) + top if top else None
        self.weight = np.zeros(model.nlight)  # the window's faded state
        self.on: list[int] = []

    def budget(self) -> int:
        return GL_LIGHTS - (1 if self.model.vis.headlight.active else 0)

    def wanted(self, x: float, y: float, top: bool = False) -> list[int]:
        """The lights to draw for a viewpoint at (x, y), most important first, at most the budget."""
        if self.model.nlight <= self.budget():
            return list(range(self.model.nlight))
        if top and self.top is not None:
            return self.top[:self.budget()]
        here = region_of(x, y)
        d = np.hypot(self.model.light_pos[:, 0] - x, self.model.light_pos[:, 1] - y)
        order = sorted(range(self.model.nlight),
                       key=lambda k: (not self.directional[k], here is None or self.region[k] != here, d[k]))
        return order[:self.budget()]

    def _apply(self, weight: np.ndarray) -> None:
        self.model.light_active[:] = weight > 0
        self.model.light_diffuse[:] = self.base * weight[:, None]

    def choose(self, x: float, y: float, top: bool = False) -> None:
        """Switch to the viewpoint's lights at once (a single image, or the robot camera)."""
        weight = np.zeros(self.model.nlight)
        weight[self.wanted(x, y, top)] = 1.0
        self._apply(weight)

    def fade(self, x: float, y: float, dt: float, top: bool = False) -> None:
        """Move the window's lights toward the viewpoint's over time: a light no longer wanted
        fades out, and a wanted one fades in once a slot is free, so the number drawn never
        exceeds the budget and nothing pops."""
        wanted = self.wanted(x, y, top)
        step = dt / FADE
        for k in list(self.on):
            if k in wanted:
                self.weight[k] = min(1.0, self.weight[k] + step)
            else:
                self.weight[k] = max(0.0, self.weight[k] - step)
                if self.weight[k] == 0.0:
                    self.on.remove(k)
        for k in wanted:
            if k not in self.on and len(self.on) < self.budget():
                self.on.append(k)
                self.weight[k] = min(1.0, step) if dt > 0 else 1.0
        self._apply(self.weight)

    def shadows(self, x: float, y: float, n: int) -> None:
        """Shadows from the n lights nearest (x, y) among those drawn that may cast them (each
        shadow-casting light re-draws the whole scene, about 1 ms)."""
        d = np.hypot(self.model.light_pos[:, 0] - x, self.model.light_pos[:, 1] - y)
        pick = [k for k in np.argsort(d) if self.shadowing[k] and self.model.light_active[k]][:n]
        self.model.light_castshadow[:] = 0
        self.model.light_castshadow[pick] = 1

    def snap(self, x: float, y: float, top: bool = False) -> None:
        """Set the window's faded state straight to the viewpoint's lights (the first frame, or
        after a view change)."""
        self.on = self.wanted(x, y, top)
        self.weight[:] = 0.0
        self.weight[self.on] = 1.0
        self._apply(self.weight)
