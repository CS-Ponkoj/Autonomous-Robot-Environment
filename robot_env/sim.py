"""MuJoCo world: physics, sensors, contact detection, and wheel motor targets.

This layer has no safety logic and no driver logic. Use RobotSystem for driving.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import mujoco
import numpy as np

from . import config as C

WORLD_XML = Path(__file__).with_name("world.xml")
_OBSTACLES_BLOCK = re.compile(r"<!-- OBSTACLES.*?/OBSTACLES -->", re.S)
_RAY_GROUPS = np.array([1, 1, 0, 1, 1, 1], dtype=np.uint8)  # skip group 2 (visual only)
_SPAWN_HEIGHT = 0.0505


def load_world_xml(include_obstacles: bool = True, extra_world_xml: str = "") -> str:
    xml = WORLD_XML.read_text(encoding="utf-8")
    if not include_obstacles:
        xml = _OBSTACLES_BLOCK.sub("", xml)
    if extra_world_xml:
        xml = xml.replace("</worldbody>", extra_world_xml + "\n  </worldbody>", 1)
    return xml


def yaw_from_quat(q: np.ndarray) -> float:
    w, x, y, z = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


class RobotSim:
    def __init__(self, include_obstacles: bool = True, extra_world_xml: str = ""):
        self.model = mujoco.MjModel.from_xml_string(load_world_xml(include_obstacles, extra_world_xml))
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
        self._ray_hit = np.zeros(1, dtype=np.int32)
        self.lidar_angles = np.array(C.LIDAR_ANGLES)
        self.lidar_fault = np.zeros(C.LIDAR_RAYS, dtype=bool)  # test hook: forced invalid rays
        self.goal = np.zeros(2)
        self._camera_renderer: mujoco.Renderer | None = None
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

    def in_contact(self) -> bool:
        n = self.data.ncon
        if n == 0:
            return False
        pairs = self.data.contact.geom[:n]
        robot = self._robot_geom[pairs]
        solid = self._solid_world_geom[pairs]
        return bool(np.any((robot[:, 0] & solid[:, 1]) | (robot[:, 1] & solid[:, 0])))

    # ----- sensors (non-privileged) -----
    def scan(self) -> tuple[np.ndarray, np.ndarray]:
        """Lidar ranges and validity. A valid ray with no return reports LIDAR_RANGE."""
        origin = self.data.site_xpos[self._lidar_site].copy()
        yaw = self.true_pose()[2]
        ranges = np.full(C.LIDAR_RAYS, C.LIDAR_RANGE)
        valid = np.ones(C.LIDAR_RAYS, dtype=bool)
        for i, angle in enumerate(self.lidar_angles):
            direction = np.array([math.cos(yaw + angle), math.sin(yaw + angle), 0.0])
            hit = mujoco.mj_ray(self.model, self.data, origin, direction, _RAY_GROUPS, 1,
                                self.robot_body, self._ray_hit)
            if 0 <= hit < C.LIDAR_RANGE:
                ranges[i] = hit
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

    def render_camera(self, size: tuple[int, int] = (240, 320)) -> np.ndarray:
        if self._camera_renderer is None or (self._camera_renderer.height, self._camera_renderer.width) != size:
            self.close()
            self._camera_renderer = mujoco.Renderer(self.model, height=size[0], width=size[1])
        self._camera_renderer.update_scene(self.data, camera="robot_cam")
        return self._camera_renderer.render()

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
            self._camera_renderer.close()
            self._camera_renderer = None
