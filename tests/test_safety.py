"""Steps 4 and 7: every safety check, in each direction where it matters."""

import math

import numpy as np
import pytest

from robot_env import config as C
from robot_env.safety import SafetyFlags, SafetyLayer, clearance_filter, is_safe
from robot_env.sim import RobotSim
from robot_env.system import RobotSystem
from robot_env.types import STOP, Command, Observation
from tests.helpers import door_edge_starts, empty_system, free_door_edge, hold, min_clearance_run

ANGLES = np.array(C.LIDAR_ANGLES)
LOW_BLOCK = '<geom name="low_block" type="box" pos="1.2 0 0.05" size="0.05 0.3 0.05" rgba="1 0 1 1"/>'


def make_obs(points=(), now=1.0, scan_time=None, invalid=(), velocity=(0.0, 0.0)):
    """Synthetic observation: the lidar sees only the given (angle_deg, range) points."""
    ranges = np.full(C.LIDAR_RAYS, C.LIDAR_RANGE)
    valid = np.ones(C.LIDAR_RAYS, dtype=bool)
    for deg, r in points:
        ranges[int(np.argmin(np.abs(np.degrees(ANGLES) - deg)))] = r
    for deg in invalid:
        i = int(np.argmin(np.abs(np.degrees(ANGLES) - deg)))
        valid[i], ranges[i] = False, 0.0
    return Observation(1, now, ranges, valid, ANGLES, now if scan_time is None else scan_time,
                       velocity, 2.0, 0.0, False)


def run(cmd, obs=None, flags=None, issued_at=0.95, now=1.0):
    return SafetyLayer().filter(cmd, issued_at, obs or make_obs(now=now), now, flags or SafetyFlags())


# ----- hard stops and their priority -----

def test_passes_safe_command_unchanged():
    r = run(Command(0.3, 0.2))
    assert r.command == Command(0.3, 0.2) and r.reasons == ()


@pytest.mark.parametrize("bad", [Command(float("nan"), 0.0), Command(0.0, float("inf"))])
def test_invalid_input_stops(bad):
    r = run(bad)
    assert r.command == STOP and r.reasons == ("invalid_input",)


def test_emergency_brake_has_priority():
    flags = SafetyFlags(emergency_brake=True, focus_lost=True, manual_mode=True)
    r = run(Command(0.3, 0.0), flags=flags)
    assert r.command == STOP and r.reasons == ("emergency_brake",)


def test_focus_lost_stops():
    assert run(Command(0.3, 0.0), flags=SafetyFlags(focus_lost=True)).reasons == ("focus_lost",)


def test_release_to_stop_in_manual_mode():
    r = run(Command(0.3, 0.0), flags=SafetyFlags(manual_mode=True, manual_input_held=False))
    assert r.command == STOP and r.reasons == ("released",)
    r = run(Command(0.3, 0.0), flags=SafetyFlags(manual_mode=True, manual_input_held=True))
    assert r.command == Command(0.3, 0.0)


def test_no_command_stops():
    assert SafetyLayer().filter(None, None, make_obs(), 1.0, SafetyFlags()).reasons == ("no_command",)


def test_command_expiry():
    assert run(Command(0.3, 0.0), issued_at=1.0 - C.COMMAND_LIFETIME - 0.01).reasons == ("command_expired",)
    assert run(Command(0.3, 0.0), issued_at=1.0 - C.COMMAND_LIFETIME + 0.01).reasons == ()


def test_stale_scan_stops():
    obs = make_obs(now=1.0, scan_time=1.0 - C.SCAN_MAX_AGE - 0.01)
    assert run(Command(0.3, 0.0), obs=obs).reasons == ("stale_scan",)


def test_speed_limits_apply_to_any_input():
    r = run(Command(5.0, -9.0))
    assert r.command == Command(C.MAX_LINEAR_SPEED, -C.MAX_ANGULAR_SPEED) and "speed_limit" in r.reasons


# ----- clearance, per direction -----

def test_forward_blocked_by_close_obstacle_ahead_but_reverse_allowed():
    obs = make_obs(points=[(0, C.FOOTPRINT_HALF_LENGTH + C.SAFETY_BUFFER)])
    assert run(Command(0.3, 0.0), obs).command == STOP
    assert run(Command(-0.15, 0.0), obs).command == Command(-0.15, 0.0)  # moving away is allowed


def test_forward_slows_down_before_stopping():
    obs = make_obs(points=[(0, C.FOOTPRINT_HALF_LENGTH + 0.12)])
    r = run(Command(0.5, 0.0), obs)
    assert 0.0 < r.command.v < 0.5 and "clearance" in r.reasons


def test_reverse_blocked_by_close_obstacle_behind_but_forward_allowed():
    obs = make_obs(points=[(180, C.FOOTPRINT_HALF_LENGTH + C.SAFETY_BUFFER)])
    assert run(Command(-0.15, 0.0), obs).command == STOP
    assert run(Command(0.3, 0.0), obs).command == Command(0.3, 0.0)


def test_far_obstacle_does_not_block():
    obs = make_obs(points=[(0, 1.2), (180, 1.2)])
    assert run(Command(0.3, 0.0), obs).command == Command(0.3, 0.0)


def test_turning_toward_a_corner_point_is_blocked_and_away_is_allowed():
    # A point 9 degrees past the front-left corner direction (41 deg), outside the buffer
    # but inside the circle swept by the corners. Turning left brings the corner to it.
    obs = make_obs(points=[(50, 0.25)])
    assert not is_safe(Command(0.0, 1.0), obs)
    assert is_safe(Command(0.0, -1.0), obs)


def test_fallback_keeps_full_turning_when_forward_blocked():
    # Ahead is too close for full forward speed, but the corners can still swing clear.
    obs = make_obs(points=[(0, C.CIRCUMSCRIBED_RADIUS + C.SAFETY_BUFFER + 0.005)])
    out = clearance_filter(Command(0.3, 1.0), obs)
    assert out.omega == 1.0 and 0.0 <= out.v < 0.3


def test_hard_stop_is_never_undone_by_fallback():
    obs = make_obs(points=[(0, 1.0)])
    r = run(Command(0.3, 1.0), obs, flags=SafetyFlags(emergency_brake=True))
    assert r.command == STOP


def test_invalid_sector_blocks_motion_into_it_only():
    obs = make_obs(invalid=[0])
    assert run(Command(0.3, 0.0), obs).command == STOP
    assert run(Command(-0.15, 0.0), obs).command == Command(-0.15, 0.0)
    obs_back = make_obs(invalid=[180])
    assert run(Command(0.3, 0.0), obs_back).command == Command(0.3, 0.0)


def test_escape_allowed_only_if_nothing_gets_closer():
    # Inside the buffer in front, rear far away: reversing out is allowed.
    obs = make_obs(points=[(0, C.FOOTPRINT_HALF_LENGTH + 0.02), (180, 1.0)])
    assert run(Command(-0.15, 0.0), obs).command == Command(-0.15, 0.0)
    # Rear also close: reversing toward it is blocked.
    obs2 = make_obs(points=[(0, C.FOOTPRINT_HALF_LENGTH + 0.02), (180, C.FOOTPRINT_HALF_LENGTH + C.SAFETY_BUFFER)])
    assert run(Command(-0.15, 0.0), obs2).command == STOP


def test_current_motion_is_included():
    # The same request near the same point: allowed when at rest, but limited when the
    # robot is already moving fast toward it (its braking travel counts).
    point = [(0, C.FOOTPRINT_HALF_LENGTH + 0.15)]
    at_rest = run(Command(0.5, 0.0), make_obs(points=point, velocity=(0.0, 0.0))).command
    moving = run(Command(0.5, 0.0), make_obs(points=point, velocity=(0.5, 0.0))).command
    assert moving.v < at_rest.v
    assert is_safe(moving, make_obs(points=point, velocity=(0.5, 0.0)))


# ----- integration through RobotSystem (real physics and lidar) -----

@pytest.mark.parametrize("v,yaw", [(0.5, 0.0), (0.3, 0.0), (-0.15, math.pi)])
def test_driving_at_a_wall_stops_before_contact(v, yaw):
    s = empty_system()
    s.reset(C.FLOOR_HALF_SIZE - 1.5, 0.0, yaw, (-2.5, -2.5))  # east wall ahead, or behind when reversing
    hold(s, v, 0.0, 12.0)
    x = s.sim.true_pose()[0]
    gap = C.FLOOR_HALF_SIZE - x - C.FOOTPRINT_HALF_LENGTH
    assert s.collisions == 0
    assert gap >= C.SAFETY_BUFFER * 0.5, gap
    assert s.intervention_events >= 1 and "clearance" in s.last_result.reasons


def test_turning_in_place_next_to_wall_never_collides():
    s = empty_system()
    y = C.FLOOR_HALF_SIZE - C.FOOTPRINT_HALF_WIDTH - 0.06  # side 6 cm from the north wall
    s.reset(0.0, y, 0.0, (-2.5, -2.5))
    for w in (1.0, -1.0):
        hold(s, 0.0, w, 3.0)
    assert s.collisions == 0


def test_driving_along_a_wall_and_into_a_corner_never_collides():
    s = empty_system()
    s.reset(0.0, C.FLOOR_HALF_SIZE - 0.3, 0.0, (-2.5, -2.5))
    hold(s, 0.5, 0.3, 10.0)  # curves toward the north wall and the corner
    assert s.collisions == 0


POLE_AT = '<geom name="thin_pole" type="cylinder" pos="1.2 {y} 0.3" size="0.012 0.3" rgba="1 0 1 1"/>'


@pytest.mark.parametrize("v", [0.3, 0.5])
@pytest.mark.parametrize("phase", [0.0, 0.5])
def test_thin_pole_is_never_hit_across_offsets_and_ray_phases(v, phase):
    """Regression (a 36-ray scan hit this pole at some offsets): a 2.4 cm pole anywhere across
    the robot's path, at two ray phases, stops the robot with the full clearance buffer."""
    for off in np.round(np.arange(-0.10, 0.1001, 0.02), 3):
        s = RobotSystem(RobotSim(include_obstacles=False, extra_world_xml=POLE_AT.format(y=off)))
        s.reset(0.0, 0.0, math.radians(phase), (-4.5, -4.5))
        low = min_clearance_run(s, s.sim.model.geom("thin_pole").id, v, 0.0, 4.0)
        assert s.collisions == 0 and low >= C.SAFETY_BUFFER, (off, low)
        s.close()


@pytest.mark.parametrize("v,w,start", [(-0.25, 0.0, (0.0, 0.04, math.pi)),   # backing into it
                                       (0.3, 0.15, (0.1, -0.12, 0.0)),       # curving into it
                                       (0.5, -0.25, (0.1, 0.15, 0.0))])
def test_thin_pole_reverse_and_curving_approaches(v, w, start):
    s = RobotSystem(RobotSim(include_obstacles=False, extra_world_xml=POLE_AT.format(y=0.0)))
    s.reset(*start, (-4.5, -4.5))
    low = min_clearance_run(s, s.sim.model.geom("thin_pole").id, v, w, 6.0)
    assert s.collisions == 0 and low >= C.SAFETY_BUFFER, low
    assert low < 0.5  # the approach really reached the pole
    s.close()


@pytest.mark.parametrize("door", ["door_office", "door_storage", "door_reception", "door_office_lab"])
def test_free_door_edges_are_never_hit_head_on(door):
    """Open door leaves are 4 cm thick: approached edge-on, both sides of the edge."""
    import mujoco
    from robot_env.layout import RoomMap
    s = RobotSystem()
    m, d = s.sim.model, s.sim.data
    mujoco.mj_forward(m, d)
    room = RoomMap(m)
    g = m.geom(door).id
    edge, out = free_door_edge(m, d, door)
    starts = list(door_edge_starts(room, edge, out))
    assert len(starts) == 3
    for off, start, yaw in starts:
        s.reset(float(start[0]), float(start[1]), yaw, (0.0, 0.0))
        low = min_clearance_run(s, g, 0.5, 0.0, 4.0)
        assert s.collisions == 0 and C.SAFETY_BUFFER <= low < 0.10, (off, low)  # reached, never hit
    s.close()


def test_obstacle_below_lidar_plane_is_recorded_by_contact_check():
    """Behind the robot neither the lidar (a plane above it) nor the depth sensor (forward only)
    sees a low obstacle; reversing into it, the separate contact check must record the hit."""
    behind = LOW_BLOCK.replace('pos="1.2 0 0.05"', 'pos="-1.2 0 0.05"')
    s = RobotSystem(RobotSim(include_obstacles=False, extra_world_xml=behind))
    s.reset(0.0, 0.0, 0.0, (-2.5, -2.5))
    hold(s, -0.3, 0.0, 8.0)
    assert s.collisions >= 1


def test_interventions_counted_as_continuous_events():
    s = empty_system()
    s.reset(C.FLOOR_HALF_SIZE - 0.8, 0.0, 0.0, (-2.5, -2.5))
    hold(s, 0.3, 0.0, 3.0)  # blocked for most of this time
    assert s.intervention_events == 1 and s.intervention_time > 1.0


def test_safety_events_are_recorded():
    s = empty_system()
    s.reset(C.FLOOR_HALF_SIZE - 0.8, 0.0, 0.0, (-2.5, -2.5))
    hold(s, 0.3, 0.0, 2.0)
    assert any("clearance" in e.reasons for e in s.safety_log)


def test_reset_zeroes_commands():
    s = empty_system()
    hold(s, 0.3, 0.0, 1.0)
    s.reset(0.0, 0.0, 0.0, (2.5, 2.5))
    assert s.requested is None
    s.advance(0.5)
    assert abs(s.sim.true_velocity()[0]) < 0.01


def test_stale_scan_stops_robot_in_system():
    s = empty_system()
    s.reset(-2.0, 0.0, 0.0, (2.5, 2.5))
    hold(s, 0.3, 0.0, 1.0)
    s.scan_enabled = False  # the lidar stops updating
    hold(s, 0.3, 0.0, 1.0)
    assert "stale_scan" in s.last_result.reasons and abs(s.sim.true_velocity()[0]) < 0.02


def test_command_expiry_stops_robot_in_system():
    s = empty_system()
    s.reset(-2.0, 0.0, 0.0, (2.5, 2.5))
    hold(s, 0.3, 0.0, 1.0)
    s.advance(1.0)  # no fresh command
    assert s.last_result.reasons == ("command_expired",) and abs(s.sim.true_velocity()[0]) < 0.02


def test_episode_over_stops():
    r = run(Command(0.3, 0.0), flags=SafetyFlags(episode_over=True))
    assert r.command == STOP and r.reasons == ("episode_over",) and not r.intervened


# ----- dropped lidar readings (5 percent dropout froze the robot) -----

def test_a_dropped_reading_takes_its_rays_recent_return_moved_by_the_robots_motion():
    from robot_env.safety import LidarMemory
    mem = LidarMemory()
    mem.fill(make_obs(points=[(0.0, 1.0)], scan_time=1.0, velocity=(0.5, 0.0)))
    out = mem.fill(make_obs(invalid=[0.0], scan_time=1.04, velocity=(0.5, 0.0)))
    i = int(np.argmin(np.abs(np.degrees(ANGLES))))
    assert out.lidar_valid[i] and out.lidar[i] == pytest.approx(1.0 - 0.5 * 0.04, abs=1e-6)


def test_a_remembered_return_expires_after_memory_age():
    from robot_env.safety import MEMORY_AGE, LidarMemory
    mem = LidarMemory()
    mem.fill(make_obs(points=[(0.0, 1.0)], scan_time=1.0))
    out = mem.fill(make_obs(invalid=[0.0], scan_time=1.0 + MEMORY_AGE + 0.02))
    assert not out.lidar_valid[int(np.argmin(np.abs(np.degrees(ANGLES))))]


def test_the_robot_keeps_moving_through_random_dropout():
    """1 m/s down the corridor with 5 percent of readings dropped at random: moving at least 90
    percent of the time, no stop longer than 0.5 s, no contact (the agreed acceptance, here in
    three seeded runs; tools runs 100)."""
    for seed in range(3):
        s = RobotSystem()
        s.faults.dropout, s.faults.seed = 0.05, seed
        s.reset(-4.3, 0.0, 0.0, (4.5, 0.0))
        moving = ticks = 0
        still = longest = 0.0
        while s.time < 6.5:
            s.drive(1.0, 0.0)
            s.advance(0.02)
            ticks += 1
            if s.observe().velocity_estimate[0] > 0.05:
                moving, still = moving + 1, 0.0
            else:
                still += 0.02
                if s.time > 1.0:
                    longest = max(longest, still)
            assert not s.in_contact
        s.close()
        assert moving / ticks >= 0.9 and longest <= 0.5


def test_a_sustained_loss_on_the_path_still_stops_the_robot():
    """A 15 degree sector straight ahead dropped for longer than the memory lasts is an unknown
    sector: driving into it is refused, turning away is not."""
    layer = SafetyLayer()
    flags = SafetyFlags()
    lost = [d for d in range(-7, 8)]
    r = None
    for k in range(10):  # 0.2 s of the same loss
        t = 1.0 + 0.02 * k
        r = layer.filter(Command(0.5, 0.0), t - 0.01, make_obs(invalid=lost, now=t, scan_time=t), t, flags)
    assert r.command.v == 0.0 and "clearance" in r.reasons
    t += 0.02
    r = layer.filter(Command(-0.2, 0.0), t - 0.01, make_obs(invalid=lost, now=t, scan_time=t), t, flags)
    assert r.command.v < 0.0


@pytest.mark.parametrize("width", [5, 15, 45])
def test_a_sustained_loss_off_the_path_does_not_stop_the_robot(width):
    """The same losses beside the robot (centred at 90 degrees) leave driving straight ahead free."""
    layer = SafetyLayer()
    lost = [90 + d for d in range(-(width // 2), width // 2 + 1)]
    r = None
    for k in range(10):
        t = 1.0 + 0.02 * k
        r = layer.filter(Command(0.5, 0.0), t - 0.01, make_obs(invalid=lost, now=t, scan_time=t), t, SafetyFlags())
    assert r.command == Command(0.5, 0.0)
