"""Step 6: lidar, camera, and contact detection."""

import math

import numpy as np
import pytest

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
    H = C.FLOOR_HALF_SIZE  # empty test track: outer walls only, inner faces at +/-H
    s.reset(1.5, 1.0, 0.0, (2.5, 2.5))
    s.advance(0.04)
    obs = s.observe()
    for deg, expected in [(0, H - 1.5), (90, H - 1.0)]:
        r, ok = ray(obs, deg)
        assert ok and abs(r - expected) <= 0.05, (deg, r, expected)
    s.reset(-2.0, -1.5, 0.0, (2.5, 2.5))
    s.advance(0.04)
    obs = s.observe()
    for deg, expected in [(-90, H - 1.5), (-180, H - 2.0)]:
        r, ok = ray(obs, deg)
        assert ok and abs(r - expected) <= 0.05, (deg, r, expected)


def test_lidar_rotates_with_robot():
    s = empty_system()
    s.reset(0.0, 2.0, math.pi / 2, (2.5, 2.5))  # facing +y, north wall 3 m ahead
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
    s.reset(-4.7, -4.7, math.pi / 4, (2.5, 2.5))  # facing the far corner, about 13 m away
    s.advance(0.04)
    r, ok = ray(s.observe(), 0)
    assert ok and r == C.LIDAR_RANGE


def test_lidar_ignores_goal_marker_and_visual_geoms():
    s = empty_system()
    s.reset(2.0, 0.0, 0.0, (3.0, 0.0))  # goal marker straight ahead at 1 m, wall at 3 m
    s.advance(0.04)
    r, ok = ray(s.observe(), 0)
    assert ok and abs(r - 3.0) <= 0.05


@pytest.mark.opengl
def test_camera_image():
    s = empty_system()
    obs = s.observe(with_camera=True)
    s.close()
    assert obs.camera.shape == (240, 320, 3) and obs.camera.dtype == np.uint8
    assert obs.camera.std() > 5


def test_contact_counted_once_per_contact_event():
    """With clearance checks off, drive into the wall: one contact event while pressed."""
    s = RobotSystem(RobotSim(include_obstacles=False), SafetyLayer(clearance_enabled=False))
    s.reset(C.FLOOR_HALF_SIZE - 0.5, 0.0, 0.0, (-2.5, -2.5))
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


@pytest.mark.opengl
def test_camera_does_not_see_through_a_close_wall():
    """Regression: a 12 cm near clip plane let the camera see through walls up close."""
    s = empty_system()
    s.reset(C.FLOOR_HALF_SIZE - C.FOOTPRINT_HALF_LENGTH - 0.03, 0.0, 0.0, (-2.5, -2.5))
    img = s.observe(with_camera=True).camera.astype(int)
    s.close()
    blue_sky = (img[:, :, 2] - img[:, :, 0]) > 40  # sky is blue-dominant, the wall is not
    assert blue_sky.mean() < 0.01


@pytest.mark.opengl
def test_camera_near_clip_plane_is_about_one_centimeter():
    s = RobotSim()
    znear = s.model.vis.map.znear * s.model.stat.extent
    assert 0.005 < znear < 0.012, znear


def _reference_scan(sim):
    """Independent per-ray reference with mj_ray (the scan itself uses one batched cast)."""
    import mujoco
    from robot_env.sim import _RAY_GROUPS
    origin = sim.data.site_xpos[sim._lidar_site].copy()
    yaw = sim.true_pose()[2]
    hit_geom = np.zeros(1, dtype=np.int32)
    out, geoms = [], []
    for a in sim.lidar_angles:
        d = np.array([math.cos(yaw + a), math.sin(yaw + a), 0.0])
        r = mujoco.mj_ray(sim.model, sim.data, origin, d, _RAY_GROUPS, 1, sim.robot_body, hit_geom)
        out.append(r if 0 <= r < C.LIDAR_RANGE else C.LIDAR_RANGE)
        geoms.append(int(hit_geom[0]) if 0 <= r < C.LIDAR_RANGE else -1)
    return np.array(out), geoms


def test_batched_scan_matches_per_ray_casts():
    """360 rays, one batched cast: identical to per-ray mj_ray at rotated poses in the furnished
    world, including no-return rays (reported as max range). Rays never stop on the robot,
    visual-only geoms (group 2), or the ceiling (group 3)."""
    sim = RobotSim()
    m = sim.model
    assert C.LIDAR_RAYS == 360 and len(sim.lidar_angles) == 360
    for x, y, yaw in [(-4.0, 0.0, 0.0), (-4.0, 0.0, 0.37), (2.5, 2.0, 2.2), (-3.0, -3.2, -1.1), (0.0, 0.0, 3.0)]:
        sim.reset(x, y, yaw, (0.0, 0.0))
        ranges, valid = sim.scan()
        ref, geoms = _reference_scan(sim)
        assert valid.all()
        assert np.allclose(ranges, ref, atol=1e-9), (x, y, yaw)
        for g in geoms:
            if g >= 0:
                assert m.geom_group[g] not in (2, 3)
                assert m.body_rootid[m.geom_bodyid[g]] != sim.robot_body
    # no return at all: an open track position facing a far direction still reports max range
    open_sim = RobotSim(include_obstacles=False)
    open_sim.reset(0.0, 0.0, 0.0, (2.0, 2.0))
    ranges, _ = open_sim.scan()
    ref, _ = _reference_scan(open_sim)
    assert np.allclose(ranges, ref) and ranges.max() <= C.LIDAR_RANGE
    # injected faults: exactly those rays are invalid (range 0); the others are unchanged
    sim.reset(-4.0, 0.0, 0.0, (0.0, 0.0))
    good, _ = sim.scan()
    sim.lidar_fault[[0, 90, 179, 359]] = True
    ranges, valid = sim.scan()
    assert not valid[[0, 90, 179, 359]].any() and (ranges[[0, 90, 179, 359]] == 0).all()
    keep = np.ones(360, dtype=bool)
    keep[[0, 90, 179, 359]] = False
    assert valid[keep].all() and np.allclose(ranges[keep], good[keep])
    sim.close()
    open_sim.close()


def test_lidar_reports_max_range_for_no_return_and_beyond_the_cutoff():
    """From the center of the open track, the diagonals have no wall within 5 m: those rays
    report exactly LIDAR_RANGE. A pole whose surface is 4.9 m away along the 45 degree ray is
    measured; one at 5.1 m (beyond the cutoff) reads as no return; one exactly at 5.0 m reads
    as LIDAR_RANGE either way."""
    def pole_at(surface):
        r = 0.012
        c = (surface + r) / math.sqrt(2)
        return f'<geom name="p" type="cylinder" pos="{c:.6f} {c:.6f} 0.3" size="{r} 0.3"/>'

    sim = RobotSim(include_obstacles=False)
    sim.reset(0.0, 0.0, 0.0, (2.0, 2.0))
    ranges, valid = sim.scan()
    diagonal = int(np.argmin(np.abs(sim.lidar_angles - math.pi / 4)))
    assert valid.all() and ranges[diagonal] == C.LIDAR_RANGE
    no_return = ranges == C.LIDAR_RANGE
    assert no_return.sum() >= 40  # four diagonal fans beyond the walls' 5 m reach
    ref, _ = _reference_scan(sim)
    assert np.array_equal(no_return, ref == C.LIDAR_RANGE)
    sim.close()
    for surface, expected in [(4.9, 4.9), (5.0, C.LIDAR_RANGE), (5.1, C.LIDAR_RANGE)]:
        sim = RobotSim(include_obstacles=False, extra_world_xml=pole_at(surface))
        sim.reset(0.0, 0.0, 0.0, (2.0, 2.0))
        origin = sim.data.site_xpos[sim._lidar_site]
        assert abs(origin[0]) < 1e-6 and abs(origin[1]) < 1e-6  # lidar at the robot center
        ranges, valid = sim.scan()
        assert valid[diagonal] and ranges[diagonal] == pytest.approx(expected, abs=1e-6), surface
        sim.close()


# ----- forward depth sensor (step Q3: obstacles below the lidar plane) -----

def _cube(size, x, y):
    h = size / 2
    return f'<geom name="cube" type="box" pos="{x} {y} {h}" size="{h} {h} {h}" rgba="1 0 1 1"/>'


def _drive_at(xml, speed, seconds=6.0, depth=True):
    """Drive straight along +x at `speed` toward whatever xml adds; returns (contacts, the least
    gap between the footprint and the cube, the stop position)."""
    from robot_env.safety import footprint_distance
    sim = RobotSim(include_obstacles=False, extra_world_xml=xml)
    s = RobotSystem(sim=sim)
    if not depth:
        sim.depth_scan = lambda: (np.full((C.DEPTH_ROWS, C.DEPTH_COLS), C.DEPTH_MAX), np.ones((C.DEPTH_ROWS, C.DEPTH_COLS), bool),
                                  np.array((0.0, 0.0, 1.0)))
    s.reset(0.0, 0.0, 0.0, (5.0, 0.0))
    g = sim.model.geom("cube").id
    cx, cy = sim.data.geom_xpos[g][:2]
    h = sim.model.geom_size[g][0]
    corners = np.array([(cx + a * h, cy + b * h) for a in np.linspace(-1, 1, 5) for b in np.linspace(-1, 1, 5)])
    least = math.inf
    while s.time < seconds:
        s.drive(speed, 0.0)
        s.advance(0.02)
        x, y, th = sim.true_pose()
        dx, dy = corners[:, 0] - x, corners[:, 1] - y
        lx, ly = dx * math.cos(th) + dy * math.sin(th), -dx * math.sin(th) + dy * math.cos(th)
        least = min(least, float(footprint_distance(lx, ly).min()))
    out = (s.collisions, least, sim.true_pose()[0])
    s.close()
    return out


@pytest.mark.parametrize("speed", [0.2, 0.3, 0.6, 1.0])
@pytest.mark.parametrize("size,offset", [(0.05, 0.0), (0.10, 0.0), (0.05, 0.1), (0.10, -0.12)])
def test_low_obstacles_below_the_lidar_plane_are_avoided(size, offset, speed):
    """5 and 10 cm cubes on the path (centred, and off-centre within the footprint) at every
    tested speed: no contact, and at least the safety buffer kept (the lidar alone sees none)."""
    contacts, least, _ = _drive_at(_cube(size, 1.6, offset), speed)
    assert contacts == 0 and least >= C.SAFETY_BUFFER - 0.005, (contacts, least)


def test_a_tall_block_stops_the_robot_no_later_than_the_lidar_alone():
    block = '<geom name="cube" type="box" pos="1.6 0 0.15" size="0.15 0.15 0.15" rgba="1 0 1 1"/>'
    _, _, with_depth = _drive_at(block, 0.6, 5.0)
    _, _, lidar_only = _drive_at(block, 0.6, 5.0, depth=False)
    assert with_depth <= lidar_only + 0.005


def test_stale_depth_allows_no_forward_motion_but_turning():
    from robot_env.safety import SafetyFlags
    from robot_env.types import Command, Observation
    ang = np.array(C.LIDAR_ANGLES)
    o = Observation(1, 1.0, np.full(C.LIDAR_RAYS, C.LIDAR_RANGE), np.ones(C.LIDAR_RAYS, bool), ang, 1.0,
                    (0.0, 0.0), 2.0, 0.0, False, None, np.full((C.DEPTH_ROWS, C.DEPTH_COLS), C.DEPTH_MAX),
                    np.ones((C.DEPTH_ROWS, C.DEPTH_COLS), bool), 1.0 - C.DEPTH_MAX_AGE - 0.02)
    r = SafetyLayer().filter(Command(0.3, 0.0), 0.99, o, 1.0, SafetyFlags())
    assert r.command.v == 0.0 and "stale_depth" in r.reasons
    r = SafetyLayer().filter(Command(0.0, 0.8), 0.99, o, 1.0, SafetyFlags())
    assert r.command.omega == 0.8


def test_depth_points_are_real_obstacles_everywhere_on_the_floor():
    """No phantom obstacle (no false stop): at poses across every room and the corridor, each
    depth point lies on a real solid (within 3 cm of one), the floor never reads as an obstacle."""
    from robot_env.layout import RoomMap, clearance, shapes_from_model
    from robot_env.sensing import depth_points
    s = RobotSystem()
    shapes = shapes_from_model(s.sim.model)
    room = RoomMap(s.sim.model)
    checked = 0
    for seed in range(40):
        task = room.sample_task(3000 + seed)
        s.reset(*task.start, task.goal)
        o = s.observe()
        pts = depth_points(o)
        if not len(pts):
            continue
        x, y, th = s.sim.true_pose()
        wx = x + pts[:, 0] * math.cos(th) - pts[:, 1] * math.sin(th)
        wy = y + pts[:, 0] * math.sin(th) + pts[:, 1] * math.cos(th)
        assert clearance(shapes, wx, wy).max() <= 0.03
        checked += len(pts)
    s.close()
    assert checked > 0
