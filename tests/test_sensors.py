"""Step 6: lidar, camera, and contact detection."""

import math

import numpy as np

from robot_env import config as C
from robot_env.safety import SafetyLayer
from robot_env.sim import RobotSim
from robot_env.system import RobotSystem
from tests.helpers import empty_system, hold


def ray(obs, degrees):
    i = int(np.argmin(np.abs(np.degrees(obs.lidar_angles) - degrees)))
    return obs.lidar[i], obs.lidar_valid[i]


def test_lidar_matches_known_wall_distances():
    s = empty_system()
    s.reset(0.5, -1.0, 0.0, (2.5, 2.5))
    s.advance(0.04)
    obs = s.observe()
    for deg, expected in [(0, 2.5), (90, 4.0), (-90, 2.0), (-180, 3.5)]:
        r, ok = ray(obs, deg)
        assert ok and abs(r - expected) <= 0.05, (deg, r, expected)


def test_lidar_rotates_with_robot():
    s = empty_system()
    s.reset(0.0, 0.0, math.pi / 2, (2.5, 2.5))  # facing +y
    s.advance(0.04)
    r, _ = ray(s.observe(), 0)
    assert abs(r - 3.0) <= 0.05


def test_no_return_is_valid_max_range_and_fault_is_invalid():
    s = RobotSystem(RobotSim(include_obstacles=False))
    s.reset(0.0, 0.0, 0.0, (2.5, 2.5))
    s.sim.lidar_fault[3] = True
    s.advance(0.04)
    obs = s.observe()
    assert not obs.lidar_valid[3] and obs.lidar[3] == 0.0
    assert obs.lidar_valid.sum() == C.LIDAR_RAYS - 1


def test_no_return_ray_is_valid_and_reports_max_range():
    s = empty_system()
    s.reset(-2.7, -2.7, math.pi / 4, (2.5, 2.5))  # facing the far corner, about 7.6 m away
    s.advance(0.04)
    r, ok = ray(s.observe(), 0)
    assert ok and r == C.LIDAR_RANGE


def test_lidar_ignores_goal_marker_and_visual_geoms():
    s = empty_system()
    s.reset(0.0, 0.0, 0.0, (1.0, 0.0))  # goal marker straight ahead at 1 m
    s.advance(0.04)
    r, ok = ray(s.observe(), 0)
    assert ok and abs(r - 3.0) <= 0.05


def test_camera_image():
    s = empty_system()
    obs = s.observe(with_camera=True)
    s.close()
    assert obs.camera.shape == (240, 320, 3) and obs.camera.dtype == np.uint8
    assert obs.camera.std() > 5


def test_contact_counted_once_per_contact_event():
    """With clearance checks off, drive into the wall: one contact event while pressed."""
    s = RobotSystem(RobotSim(include_obstacles=False), SafetyLayer(clearance_enabled=False))
    s.reset(2.5, 0.0, 0.0, (-2.5, -2.5))
    hold(s, 0.3, 0.0, 3.0)
    assert s.collisions == 1 and s.observe().contact
    hold(s, -0.15, 0.0, 1.5)  # back away, then hit again
    assert not s.observe().contact
    hold(s, 0.3, 0.0, 2.0)
    assert s.collisions == 2


def test_floor_and_goal_marker_are_not_contacts():
    s = empty_system()
    s.reset(0.0, 0.0, 0.0, (0.5, 0.0))
    hold(s, 0.3, 0.0, 2.0)  # drive over the goal disc
    assert s.collisions == 0


def test_camera_does_not_see_through_a_close_wall():
    """Regression: a 12 cm near clip plane let the camera see through walls up close."""
    s = empty_system()
    s.reset(C.ROOM_HALF_SIZE - C.FOOTPRINT_HALF_LENGTH - 0.03, 0.0, 0.0, (-2.5, -2.5))
    img = s.observe(with_camera=True).camera.astype(int)
    s.close()
    blue_sky = (img[:, :, 2] - img[:, :, 0]) > 40  # sky is blue-dominant, the wall is not
    assert blue_sky.mean() < 0.01
