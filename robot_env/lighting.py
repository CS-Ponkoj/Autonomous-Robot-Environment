"""Which lights the renderer draws. The classic renderer draws at most eight lights (the headlight
one of them), the first ones in model order, so a world with more shows only its first few
everywhere. Before each render, the lights that matter to the viewpoint are switched on and the
rest off: directional lights always, then the room lights (those that may cast shadows) of the
room the eye is in and of the room it looks into, then their window lights, then the nearest
others. One Lights per model, made before anything renders (it reads the world's own shadow
settings, which shadows() then changes). Visual only: physics, sensors, and contacts never read
lights."""

from __future__ import annotations

import math

import mujoco
import numpy as np

from . import config as C
from .layout import region_of

GL_LIGHTS = 8  # the most the classic renderer draws, the headlight included
ROOM_FITTINGS = 2  # a room's nearest fittings drawn before the next room's
DOOR_AHEAD = 4.0  # m: a room whose doorway is ahead of the eye this near gets its fitting nearest the door
DOOR_CONE = math.radians(60)  # how far off the view a doorway still counts as ahead
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

    def wanted(self, x: float, y: float, top: bool = False, look=None) -> list[int]:
        """The lights to draw for an eye at (x, y) looking toward `look` (x, y; the eye itself if
        None), most important first, at most the budget. An eye in a doorway (in no room) takes
        the room it looks into."""
        if self.model.nlight <= self.budget():
            return list(range(self.model.nlight))
        if top and self.top is not None:
            return self.top[:self.budget()]
        lx, ly = (x, y) if look is None else look
        seen = region_of(lx, ly)
        here = region_of(x, y) or seen
        d = np.hypot(self.model.light_pos[:, 0] - x, self.model.light_pos[:, 1] - y)
        near = sorted(range(self.model.nlight), key=lambda k: d[k])

        def room(region, fitting):
            return [k for k in near if not self.directional[k] and self.shadowing[k] == fitting
                    and region is not None and self.region[k] == region]
        own, other = room(here, True), (room(seen, True) if seen != here else [])
        order = list(np.flatnonzero(self.directional))
        windows = room(here, False) + (room(seen, False) if seen != here else [])
        # of each room seen through a doorway ahead, the fitting nearest that doorway
        through = [min(room(r, True), key=lambda k: math.hypot(self.model.light_pos[k, 0] - dx, self.model.light_pos[k, 1] - dy))
                   for r, (dx, dy) in self._rooms_through_doors(x, y, lx, ly, here) if room(r, True)]
        # each room's two nearest fittings first (a corridor has four), one of each room seen
        # through a doorway ahead, the room looked into, then the rest, then windows
        for k in own[:ROOM_FITTINGS] + through + other[:ROOM_FITTINGS] + own + other + windows + near:
            if k not in order:
                order.append(k)
        return order[:self.budget()]

    @staticmethod
    def _rooms_through_doors(x, y, lx, ly, here) -> list:
        """(room, doorway) for the rooms on the far side of the doorways ahead of an eye at (x, y)
        looking toward (lx, ly): within DOOR_AHEAD and DOOR_CONE of the view, nearest first."""
        fx, fy = lx - x, ly - y
        f = math.hypot(fx, fy)
        if f < 1e-6:
            return []
        found = []
        for dx, dy in sorted(C.DOORS, key=lambda dd: math.hypot(dd[0] - x, dd[1] - y)):
            r = math.hypot(dx - x, dy - y)
            if r > DOOR_AHEAD or (dx - x) * fx + (dy - y) * fy <= r * f * math.cos(DOOR_CONE):
                continue
            for ox, oy in ((0.0, 0.3), (0.0, -0.3), (0.3, 0.0), (-0.3, 0.0)):
                room = region_of(dx + ox, dy + oy)
                if room is not None and room != here and room not in [f for f, _ in found]:
                    found.append((room, (dx, dy)))
        return found

    def _apply(self, weight: np.ndarray) -> None:
        self.model.light_active[:] = weight > 0
        self.model.light_diffuse[:] = self.base * weight[:, None]

    def choose(self, x: float, y: float, top: bool = False, look=None) -> None:
        """Switch to the viewpoint's lights at once (a single image, or the robot camera)."""
        weight = np.zeros(self.model.nlight)
        weight[self.wanted(x, y, top, look)] = 1.0
        self._apply(weight)

    def fade(self, x: float, y: float, dt: float, top: bool = False, look=None) -> None:
        """Move the window's lights toward the viewpoint's over time: a light no longer wanted
        fades out, and a wanted one fades in once a slot is free, so the number drawn never
        exceeds the budget and nothing pops."""
        wanted = self.wanted(x, y, top, look)
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
        """Shadows from n of the drawn lights that may cast them: those of the room (x, y) is in
        first, then the nearest (each shadow-casting light re-draws the whole scene, about 1 ms)."""
        here = region_of(x, y)
        d = np.hypot(self.model.light_pos[:, 0] - x, self.model.light_pos[:, 1] - y)
        order = sorted(range(self.model.nlight), key=lambda k: (here is None or self.region[k] != here, d[k]))
        pick = [k for k in order if self.shadowing[k] and self.model.light_active[k]][:n]
        self.model.light_castshadow[:] = 0
        self.model.light_castshadow[pick] = 1

    def snap(self, x: float, y: float, top: bool = False, look=None) -> None:
        """Set the window's faded state straight to the viewpoint's lights (the first frame, or
        after a view change)."""
        self.on = self.wanted(x, y, top, look)
        self.weight[:] = 0.0
        self.weight[self.on] = 1.0
        self._apply(self.weight)
