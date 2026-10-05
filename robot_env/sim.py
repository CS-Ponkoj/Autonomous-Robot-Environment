"""MuJoCo world: physics, sensors, contact detection, and wheel motor targets.

This layer has no safety logic and no driver logic. Use RobotSystem for driving.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
import re
from pathlib import Path

import mujoco
import numpy as np

from . import config as C
from . import kernels

WORLD_XML = Path(__file__).with_name("world.xml")
ASSETS = Path(__file__).with_name("assets")
_OBSTACLES_BLOCK = re.compile(r"<!-- OBSTACLES.*?/OBSTACLES -->", re.S)
_RAY_GROUPS = np.array([1, 1, 0, 0, 1, 1], dtype=np.uint8)  # skip visual (2) and ceiling (3)
_WORLD_SOLID_GROUP = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)  # world solids only (tires are group 1)
_VIEW_GROUP = np.array([1, 0, 1, 0, 0, 1], dtype=np.uint8)  # what blocks the viewer: solids, visual detail, cats
_SIGHT_GROUP = np.array([1, 0, 0, 0, 1, 0], dtype=np.uint8)  # what blocks a cat's view: walls and furniture
_SPAWN_HEIGHT = 0.0505


def close_renderer(renderer: mujoco.Renderer) -> None:
    """Free a renderer's OpenGL objects in its own context, then the context. (Renderer.close
    frees the context first and then the objects in whatever context is current: with two
    renderers in a process, closing one deleted the other's textures and buffers, and that view
    went black: the window's view after the robot-camera preview changed size.)"""
    gl = getattr(renderer, "_gl_context", None)
    mjr = getattr(renderer, "_mjr_context", None)
    if gl is not None and mjr is not None:
        gl.make_current()
        mjr.free()
        renderer._mjr_context = None  # freed: close() frees only the context now
    renderer.close()


@dataclass(frozen=True)
class WorldExtra:
    """Additions to the world model: assets, bodies, skins, and in-memory files they use."""
    asset: str = ""
    worldbody: str = ""
    deformable: str = ""
    files: dict = field(default_factory=dict)


def load_world_xml(include_obstacles: bool = True, extra_world_xml: str = "", extra: WorldExtra | None = None) -> str:
    xml = WORLD_XML.read_text(encoding="utf-8")
    if not include_obstacles:
        xml = _OBSTACLES_BLOCK.sub("", xml)
    body = extra_world_xml + (extra.worldbody if extra else "")
    if body:
        xml = xml.replace("</worldbody>", body + "\n  </worldbody>", 1)
    if extra and extra.asset:
        xml = xml.replace("</asset>", extra.asset + "\n  </asset>", 1)
    if extra and extra.deformable:
        xml = xml.replace("</mujoco>", f"  <deformable>{extra.deformable}</deformable>\n</mujoco>", 1)
    return xml


def yaw_from_quat(q: np.ndarray) -> float:
    w, x, y, z = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


class RobotSim:
    def __init__(self, include_obstacles: bool = True, extra_world_xml: str = "", extra: WorldExtra | None = None):
        assets = {p.name: p.read_bytes() for p in ASSETS.glob("*.png")}
        if extra is not None:
            assets.update(extra.files)
        self.model = mujoco.MjModel.from_xml_string(load_world_xml(include_obstacles, extra_world_xml, extra), assets)
        self.pre_render: list = []  # callables run before any render (e.g. posing the cats' skins)
        if abs(self.model.opt.timestep - C.PHYSICS_DT) > 1e-12:
            raise ValueError("world.xml timestep must match config.PHYSICS_DT")
        self.data = mujoco.MjData(self.model)
        m = self.model
        self.robot_body = m.body("robot").id
        self._lidar_site = m.site("lidar_site").id
        self._left_motor = m.actuator("left_motor").id
        self._right_motor = m.actuator("right_motor").id
        self._left_dof = m.joint("left_wheel_joint").dofadr[0]
        self._right_dof = m.joint("right_wheel_joint").dofadr[0]
        self._goal_mocap = m.body_mocapid[m.body("goal").id]
        self._floor = m.geom("floor").id
        roots = m.body_rootid[m.geom_bodyid]
        self._robot_geom = roots == self.robot_body
        self._solid_world_geom = (~self._robot_geom) & (m.geom_contype != 0) & (np.arange(m.ngeom) != self._floor)
        self._cat_geom = np.array([m.body(int(b)).name.startswith("cat") for b in m.geom_bodyid], dtype=bool)
        self._ray_hit = np.zeros(1, dtype=np.int32)
        self._scan_geom = np.zeros(C.LIDAR_RAYS, dtype=np.int32)
        self._scan_dist = np.zeros(C.LIDAR_RAYS)
        self._scan_dirs = np.zeros((C.LIDAR_RAYS, 3))
        self.lidar_angles = np.array(C.LIDAR_ANGLES)
        self.lidar_angles.flags.writeable = False  # the scanner's geometry: never changed
        self.lidar_fault = np.zeros(C.LIDAR_RAYS, dtype=bool)  # test hook: forced invalid rays
        self.goal = np.zeros(2)
        self._camera_renderer: mujoco.Renderer | None = None
        self.camera_option = mujoco.MjvOption()
        self.camera_option.geomgroup[3] = 1  # the robot camera sees the ceiling
        self.reset(0.0, 0.0, 0.0, (2.0, 2.0))

    # ----- state -----
    def reset(self, x: float, y: float, yaw: float, goal_xy: tuple[float, float]) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[0:3] = (x, y, _SPAWN_HEIGHT)
        self.data.qpos[3:7] = (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))
        self.goal = np.array(goal_xy, dtype=float)
        self.data.mocap_pos[self._goal_mocap] = (goal_xy[0], goal_xy[1], 0.0)
        self.lidar_fault[:] = False
        mujoco.mj_forward(self.model, self.data)

    @property
    def time(self) -> float:
        return float(self.data.time)

    def set_wheel_targets(self, v: float, omega: float) -> None:
        """Raw motor targets with no safety checks. Only RobotSystem should call this."""
        half = C.WHEEL_SEPARATION / 2
        self.data.ctrl[self._left_motor] = (v - omega * half) / C.WHEEL_RADIUS
        self.data.ctrl[self._right_motor] = (v + omega * half) / C.WHEEL_RADIUS

    def physics_step(self) -> bool:
        """Advance one physics step. Returns True if the robot touches a wall or obstacle."""
        mujoco.mj_step(self.model, self.data)
        return self.in_contact()

    def _robot_contacts(self) -> tuple[bool, bool]:
        """(the robot touches something solid, the robot touches a cat), now."""
        n = self.data.ncon
        if n == 0:
            return False, False
        return kernels.robot_contacts(self.data.contact.geom[:n], self._robot_geom, self._solid_world_geom,
                                      self._cat_geom)

    def in_contact(self) -> bool:
        return self._robot_contacts()[0]

    def cat_contacts_now(self) -> dict[int, list[tuple[int, float]]]:
        """Every cat the robot touches now: {cat index: [(contact index, +1 or -1), ...]}. The sign
        turns the contact normal (geom1 -> geom2) into the robot -> cat direction."""
        out: dict[int, list[tuple[int, float]]] = {}
        if not self._robot_contacts()[1]:
            return out  # (the usual case: no cat touches the robot)
        for c in range(self.data.ncon):
            g1, g2 = self.data.contact.geom[c]
            for a, b, sign in ((g1, g2, 1.0), (g2, g1, -1.0)):
                if self._robot_geom[a] and self._cat_geom[b]:
                    index = int(self.model.body(int(self.model.geom_bodyid[b])).name[3:].split("_")[0])
                    out.setdefault(index, []).append((c, sign))
        return out

    # ----- sensors (non-privileged) -----
    def scan(self) -> tuple[np.ndarray, np.ndarray]:
        """Lidar ranges and validity. A valid ray with no return reports LIDAR_RANGE."""
        origin = self.data.site_xpos[self._lidar_site].copy()
        yaw = self.true_pose()[2]
        ranges = np.full(C.LIDAR_RAYS, C.LIDAR_RANGE)
        valid = np.ones(C.LIDAR_RAYS, dtype=bool)
        directions = self._scan_dirs  # preallocated workspace (z stays 0: one horizontal plane)
        directions[:, 0] = np.cos(yaw + self.lidar_angles)
        directions[:, 1] = np.sin(yaw + self.lidar_angles)
        # One batched cast (same results as mj_ray per ray within LIDAR_RANGE; geoms beyond it
        # are ignored, which reads as "no return").
        mujoco.mj_multiRay(self.model, self.data, origin, directions.ravel(), _RAY_GROUPS, 1,
                           self.robot_body, self._scan_geom, self._scan_dist, None, C.LIDAR_RAYS, C.LIDAR_RANGE)
        hit = (self._scan_dist >= 0) & (self._scan_dist < C.LIDAR_RANGE)
        ranges[hit] = self._scan_dist[hit]
        bad = self.lidar_fault | ~np.isfinite(ranges)
        valid[bad] = False
        ranges[bad] = 0.0
        return ranges, valid

    def velocity_estimate(self) -> tuple[float, float]:
        """(v, omega) from wheel encoders. Differs from true motion when wheels slip."""
        wl = self.data.qvel[self._left_dof] * C.WHEEL_RADIUS
        wr = self.data.qvel[self._right_dof] * C.WHEEL_RADIUS
        return (float((wl + wr) / 2), float((wr - wl) / C.WHEEL_SEPARATION))

    def goal_sensor(self) -> tuple[float, float]:
        """Ideal goal sensor (privileged simulation assumption): distance and bearing."""
        x, y, yaw = self.true_pose()
        dx, dy = self.goal[0] - x, self.goal[1] - y
        bearing = math.atan2(dy, dx) - yaw
        return math.hypot(dx, dy), math.atan2(math.sin(bearing), math.cos(bearing))

    def before_render(self) -> None:
        """Bring visual-only state up to date before drawing (the cats' skinned meshes): run the
        hooks, then propagate the kinematics, so the skin is drawn from the bone poses just
        written (not from the last physics step). Physics is unaffected: every step recomputes
        the kinematics from the same inputs."""
        if not self.pre_render:
            return
        for hook in self.pre_render:
            hook()
        mujoco.mj_kinematics(self.model, self.data)

    def render_camera(self, size: tuple[int, int] = (240, 320), posed: bool = False,
                      reflections: bool = True) -> np.ndarray:
        """The robot camera's image. posed=True: the caller has just run before_render at this
        simulated time (the same frame), so the skins are not posed a second time.
        reflections=False skips the floor's reflection pass (the window's small preview)."""
        self.prepare_camera(size, posed, reflections)
        return self.render_prepared()

    def prepare_camera(self, size: tuple[int, int] = (240, 320), posed: bool = False,
                       reflections: bool = True) -> None:
        """The first half of render_camera: the robot camera's scene as it is now (lights,
        shadows, and skins included), drawn by render_prepared (the window splits the two over
        consecutive frames)."""
        if self._camera_renderer is None or (self._camera_renderer.height, self._camera_renderer.width) != size:
            self.close()
            self._camera_renderer = mujoco.Renderer(self.model, height=size[0], width=size[1])
        if not posed:
            self.before_render()
        self._camera_renderer.update_scene(self.data, camera="robot_cam", scene_option=self.camera_option)
        self._camera_renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = reflections

    def render_prepared(self) -> np.ndarray:
        """The second half of render_camera: draw the scene prepare_camera made."""
        return self._camera_renderer.render()

    def ray_to_solid(self, origin, direction) -> float:
        """Distance to the first solid world geom along a unit direction (-1 if none).
        Used by the viewer camera only; ignores the robot, visuals, and the ceiling."""
        return float(mujoco.mj_ray(self.model, self.data, np.asarray(origin, float), np.asarray(direction, float),
                                   _WORLD_SOLID_GROUP, 1, self.robot_body, self._ray_hit))

    def sees_robot(self, origin) -> bool:
        """Whether the robot is in plain sight from a point (a cat's eyes): no wall or furniture
        between them (the cats are world actors; this is not a sensor of the robot's)."""
        x, y, _ = self.true_pose()
        target = np.array([x, y, 0.15])
        d = target - np.asarray(origin, float)
        dist = float(np.linalg.norm(d))
        if dist < 1e-9:
            return True
        hit = float(mujoco.mj_ray(self.model, self.data, np.asarray(origin, float), d / dist, _SIGHT_GROUP, 1,
                                  self.robot_body, self._ray_hit))
        return hit < 0 or hit > dist - C.CIRCUMSCRIBED_RADIUS - 0.05

    def ray_to_view_blocker(self, origin, direction) -> float:
        """Like ray_to_solid, but visual-only detail (chair arms, decor) also counts: anything
        that would block the viewer's line of sight. Viewer camera only."""
        return float(mujoco.mj_ray(self.model, self.data, np.asarray(origin, float), np.asarray(direction, float),
                                   _VIEW_GROUP, 1, self.robot_body, self._ray_hit))

    def rays_to_view_blocker(self, origin, directions: np.ndarray, cutoff: float = mujoco.mjMAXVAL) -> np.ndarray:
        """ray_to_view_blocker for many directions from one origin in a single call (-1: nothing
        within `cutoff` metres; a short cutoff makes the call much cheaper)."""
        n = len(directions)
        dist = np.empty(n)
        geom = np.empty(n, dtype=np.int32)
        mujoco.mj_multiRay(self.model, self.data, np.asarray(origin, float), np.ascontiguousarray(directions, float).ravel(),
                           _VIEW_GROUP, 1, self.robot_body, geom, dist, None, n, cutoff)
        return dist

    # ----- ground truth (evaluation only, never given to drivers) -----
    def true_pose(self) -> tuple[float, float, float]:
        q = self.data.qpos
        return float(q[0]), float(q[1]), yaw_from_quat(q[3:7])

    def true_velocity(self) -> tuple[float, float]:
        yaw = self.true_pose()[2]
        qv = self.data.qvel
        return float(qv[0] * math.cos(yaw) + qv[1] * math.sin(yaw)), float(qv[5])

    def close(self) -> None:
        if self._camera_renderer is not None:
            close_renderer(self._camera_renderer)
            self._camera_renderer = None
