"""Shared test helpers. The waypoint follower is TEST-ONLY: it reads the true pose
and the planning map, so it shows the goal is reachable through the shared drive
and safety path. It is not evidence of autonomous perception or manual usability."""

from __future__ import annotations

import math

from robot_env import config as C
from robot_env.layout import Task
from robot_env.sim import RobotSim
from robot_env.system import RobotSystem
from robot_env.types import Command, Decision, Observation


def empty_system() -> RobotSystem:
    return RobotSystem(RobotSim(include_obstacles=False))


def hold(system: RobotSystem, v: float, w: float, seconds: float, period: float = C.DECISION_PERIOD) -> None:
    """Send the same command at the decision rate for `seconds`."""
    for _ in range(int(round(seconds / period))):
        system.drive(v, w)
        system.advance(period)


class WaypointFollower:
    """Test-only driver that follows the planned path using the true pose."""

    name = "test_waypoint_follower"

    def __init__(self, system: RobotSystem, task: Task):
        self.system = system
        self.points = list(task.path[1:])

    def command(self) -> tuple[float, float]:
        x, y, yaw = self.system.sim.true_pose()
        while len(self.points) > 1 and math.dist((x, y), self.points[0]) < 0.2:
            self.points.pop(0)
        tx, ty = self.points[0]
        err = math.atan2(ty - y, tx - x) - yaw
        err = math.atan2(math.sin(err), math.cos(err))
        if abs(err) > 0.5:
            return 0.0, math.copysign(1.0, err)
        return 0.3, max(-1.0, min(1.0, 2.5 * err))

    def decide(self, obs: Observation) -> Decision:
        """Same decide(observation) plug as every driver (this one also reads the true pose)."""
        v, w = self.command()
        return Decision(Command(v, w), obs.seq)
