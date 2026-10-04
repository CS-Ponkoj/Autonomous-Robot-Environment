"""Wandering cats (off by default): geometry, motion bounds, sensing, contacts, determinism."""

import math

import mujoco
import numpy as np
import pytest

from robot_env import config as C
from robot_env.cats import (ACCEL, CAT_GAP, MAX_CATS, ROBOT_GAP, V_MAX, WALL_GAP, YAW_ACCEL, YAW_RATE,
                            cat_xml)
from robot_env.sim import RobotSim
from robot_env.system import RobotSystem, mujoco_forward


def test_cats_off_keeps_the_default_world_unchanged():
    """With no cats the system loads the plain world (no cat bodies at all)."""
    plain = RobotSim()
    s = RobotSystem()
    assert s.cats is None and s.sim.model.nbody == plain.model.nbody and s.sim.model.ngeom == plain.model.ngeom
    assert not any(s.sim.model.body(i).name.startswith("cat") for i in range(s.sim.model.nbody))
    s.close()
    plain.close()


def test_cat_count_and_seed_are_validated():
    assert cat_xml(MAX_CATS) and cat_xml(0) == ""
    with pytest.raises(ValueError):
        cat_xml(MAX_CATS + 1)
    with pytest.raises(ValueError):
        RobotSystem(cats=-1)
    with pytest.raises(ValueError):
        RobotSystem(cats=1, cat_seed=-3)
    with pytest.raises(ValueError):
        RobotSystem(RobotSim(), cats=1)  # cats need the system's own cat-enabled world


def _run(seed=4, cats=3, seconds=40.0, start=(0.0, 0.0, 0.0), goal=(4.0, 4.0), robot_cmd=(0.0, 0.0), each=None):
    s = RobotSystem(cats=cats, cat_seed=seed)
    s.reset(*start, goal)
    for i in range(int(round(seconds / C.PHYSICS_DT))):
        if i % 10 == 0:
            s.drive(*robot_cmd)
        s.advance(C.PHYSICS_DT)
        if each:
            each(s, i)
    return s


def test_cats_never_overlap_furniture_robot_or_each_other_and_respect_motion_bounds():
    """A long seeded run: every physics step the cats keep their gaps, never leave the floor,
    and speed, acceleration, and turn rate stay inside their bounds; no solver warnings."""
    prev = {}
    worst = {"accel": 0.0, "yaw_rate": 0.0, "yaw_accel": 0.0, "speed": 0.0, "step": 0.0}

    def check(s, i):
        for cat in s.cats.cats:
            wall, robot, other = s.cats.gaps(cat, cat.x, cat.y, cat.yaw)
            assert wall >= WALL_GAP - 1e-9 and robot >= ROBOT_GAP - 1e-9 and other >= CAT_GAP - 1e-9, (i, cat)
            assert abs(cat.x) < C.FLOOR_HALF_SIZE and abs(cat.y) < C.FLOOR_HALF_SIZE
            if cat.index in prev:
                px, py, pyaw, pv, pw = prev[cat.index]
                worst["step"] = max(worst["step"], math.hypot(cat.x - px, cat.y - py))
                worst["accel"] = max(worst["accel"], abs(cat.v - pv) / C.PHYSICS_DT)
                worst["yaw_accel"] = max(worst["yaw_accel"], abs(cat.w - pw) / C.PHYSICS_DT)
            worst["speed"] = max(worst["speed"], abs(cat.v))
            worst["yaw_rate"] = max(worst["yaw_rate"], abs(cat.w))
            prev[cat.index] = (cat.x, cat.y, cat.yaw, cat.v, cat.w)

    s = _run(each=check)
    assert worst["speed"] <= V_MAX + 1e-9 and worst["yaw_rate"] <= YAW_RATE + 1e-9
    assert worst["accel"] <= ACCEL + 1e-6 and worst["yaw_accel"] <= YAW_ACCEL + 1e-6
    assert worst["step"] <= V_MAX * C.PHYSICS_DT + 1e-9  # continuous motion: no jumps
    assert sum(int(w.number) for w in s.sim.data.warning) == 0
    states = {st for cat in s.cats.cats for _, st in cat.transitions}
    assert {"walk", "pause"} <= states  # the behavior really varies
    s.close()


def test_cat_collision_bodies_stay_clear_of_static_geometry_exactly():
    """Exact MuJoCo distance from every cat collision geom to every static solid stays positive
    (checked every 10 ms of a long run), so the clearance raster is conservative."""
    hits = []

    def check(s, i):
        if i % 20:
            return
        m, d = s.sim.model, s.sim.data
        cat_geoms = [g for g in range(m.ngeom) if m.body(int(m.geom_bodyid[g])).name.startswith("cat")
                     and m.geom_contype[g]]
        statics = np.flatnonzero(s.sim._solid_world_geom & ~s.sim._cat_geom)
        for g in cat_geoms:
            for w in statics:
                dist = mujoco.mj_geomDistance(m, d, g, int(w), 0.05, None)
                if dist < 0.0:
                    hits.append((i, m.geom(g).name, m.geom(int(w)).name, dist))

    s = _run(seed=9, seconds=30.0, each=check)
    assert not hits, hits[:5]
    s.close()


def test_cats_are_deterministic_per_seed():
    def track(seed):
        s = _run(seed=seed, cats=2, seconds=8.0)
        out = [(c.x, c.y, c.yaw, c.state) for c in s.cats.cats]
        s.close()
        return out
    assert track(5) == track(5)
    assert track(5) != track(6)


def test_lidar_sees_a_cat_standing_sitting_and_turned():
    """The cat's collision body crosses the lidar plane in every pose: the scan returns a hit at
    the cat's bearing at about the cat's distance, across distances and between-ray offsets."""
    s = RobotSystem(cats=1, cat_seed=0)
    s.reset(-2.0, -0.0, 0.0, (4.0, 4.0))  # open corridor, robot facing +x
    cat = s.cats.cats[0]
    misses = []
    for dist in (0.6, 1.2, 2.0, 3.0):
        for offset_deg in (0.0, 0.3, 0.5):
            for yaw, pitch in ((0.0, 0.0), (math.pi / 2, 0.0), (math.pi / 4, math.radians(35))):
                bearing = math.radians(offset_deg)
                cat.x, cat.y = -2.0 + dist * math.cos(bearing), dist * math.sin(bearing)
                cat.yaw, cat.pitch = yaw, pitch
                s.cats._write_poses()
                mujoco_forward(s.sim)
                ranges, valid = s.sim.scan()
                near = np.abs(np.degrees(s.sim.lidar_angles) - offset_deg) <= 3.0
                # the ray hits a cat geom (not something behind it) at about the cat's distance
                seen = (near & valid & _cat_rays(s) & (ranges > dist - 0.25) & (ranges < dist + 0.1)).any()
                if not seen:
                    misses.append((dist, offset_deg, round(yaw, 2), round(math.degrees(pitch))))
    assert not misses, misses
    s.close()


def _cat_rays(s):
    m = s.sim.model
    return np.array([g >= 0 and m.body(int(m.geom_bodyid[g])).name.startswith("cat") for g in s.sim._scan_geom])


def _inject(s, poses):
    """Fault injection (the cat policy never does this): place cats and drive them straight,
    bypassing the policy's gaps and behavior."""
    s.cats.pose_ok = lambda *a: True
    s.cats.tick = lambda: None
    for cat, (x, y, yaw, v) in zip(s.cats.cats, poses):
        cat.x, cat.y, cat.yaw, cat.v, cat.target_v, cat.target_yaw = x, y, yaw, v, v, yaw
    s.cats._write_poses()
    mujoco_forward(s.sim)


def test_a_cat_walking_into_a_stopped_robot_is_a_collision_and_a_cat_contact():
    """A cat driven into the stationary robot counts as a collision and a cat contact with the
    cat as initiator; the cat freezes on contact, so the robot is never shoved or tunneled into;
    force and impulse are recorded."""
    s = RobotSystem(cats=1, cat_seed=0)
    s.reset(-2.0, 0.0, 0.0, (4.0, 4.0))
    _inject(s, [(-1.2, 0.0, math.pi, 0.4)])
    x0, y0, _ = s.sim.true_pose()
    worst_v, worst_shift = 0.0, 0.0
    for _ in range(int(3.0 / C.PHYSICS_DT)):
        s.advance(C.PHYSICS_DT)
        x, y, _ = s.sim.true_pose()
        worst_v = max(worst_v, abs(s.sim.true_velocity()[0]))
        worst_shift = max(worst_shift, math.hypot(x - x0, y - y0))
    assert s.collisions == 1 and s.cat_contacts == 1
    event = s.cat_contact_events[0]
    assert event["cat"] == 0 and event["initiator"] == "cat" and event["cat_closing_speed"] > 0.1
    assert event["robot_closing_speed"] < 0.02
    assert event["peak_force_n"] > 0.0 and event["impulse_ns"] > 0.0
    cat = s.cats.cats[0]
    assert cat.frozen and cat.v == 0.0
    assert worst_shift < 0.01 and worst_v < 0.05, (worst_shift, worst_v)  # bounded: no shove
    assert cat.x - s.sim.true_pose()[0] > 0.2  # no tunneling: the cat stays in front of the robot
    s.close()


def test_a_robot_driving_into_a_still_cat_is_the_robot_initiator():
    """With the clearance filter off (fault injection), the robot drives into a still cat."""
    from robot_env.safety import SafetyLayer
    s = RobotSystem(safety=SafetyLayer(clearance_enabled=False), cats=1, cat_seed=0)
    s.reset(-2.0, 0.0, 0.0, (4.0, 4.0))
    _inject(s, [(-1.3, 0.0, math.pi / 2, 0.0)])
    for i in range(int(4.0 / C.PHYSICS_DT)):
        if i % 10 == 0:
            s.drive(0.3, 0.0)
        s.advance(C.PHYSICS_DT)
        if s.cat_contacts:
            break
    assert s.cat_contacts == 1 and s.collisions == 1
    event = s.cat_contact_events[0]
    assert event["initiator"] == "robot" and event["robot_closing_speed"] > 0.1, event
    s.close()


def test_two_cats_touching_the_robot_are_two_contacts_and_two_collisions():
    """Per-cat contact tracking is independent of the global collision debounce: a second cat
    arriving while the first still touches is still counted."""
    s = RobotSystem(cats=2, cat_seed=0)
    s.reset(-2.0, 0.0, 0.0, (4.0, 4.0))
    _inject(s, [(-1.25, 0.0, math.pi, 0.3), (-2.0, -0.9, math.pi / 2, 0.3)])
    for _ in range(int(4.0 / C.PHYSICS_DT)):
        s.advance(C.PHYSICS_DT)
    assert s.cat_contacts == 2 and s.collisions == 2, s.cat_contact_events
    assert sorted(e["cat"] for e in s.cat_contact_events) == [0, 1]
    assert s.cat_contact_events[1]["t"] - s.cat_contact_events[0]["t"] > C.PHYSICS_DT
    s.close()


def test_two_cats_touching_on_the_same_step_are_two_collisions():
    """Mirror-image cats reach the robot on the same physics step: both count."""
    s = RobotSystem(cats=2, cat_seed=0)
    s.reset(-2.0, 0.0, 0.0, (4.0, 4.0))
    _inject(s, [(-2.0, 0.9, -math.pi / 2, 0.3), (-2.0, -0.9, math.pi / 2, 0.3)])
    for _ in range(int(4.0 / C.PHYSICS_DT)):
        s.advance(C.PHYSICS_DT)
        if s.cat_contacts:
            break
    assert s.cat_contacts == 2 and s.collisions == 2, s.cat_contact_events
    assert s.cat_contact_events[0]["t"] == s.cat_contact_events[1]["t"]
    s.close()


def _cat_events(path):
    from robot_env.drive_log import read_log
    return [r for r in read_log(path) if r["kind"] == "event" and r["name"].startswith("cat_contact")]


def test_a_contact_still_open_at_close_is_finished_in_the_log(tmp_path):
    """The robot and the frozen cat stay touching; closing the system ends the contact in the
    log with its force, impulse, duration, and reason."""
    from robot_env.drive_log import DriveLog
    s = RobotSystem(cats=1, cat_seed=0)
    s.log = DriveLog(tmp_path / "open.jsonl")
    s.reset(-2.0, 0.0, 0.0, (4.0, 4.0))
    _inject(s, [(-1.2, 0.0, math.pi, 0.4)])
    while not s.cat_contacts:
        s.advance(C.PHYSICS_DT)
    s.advance(0.05)
    assert s._cat_open  # still touching
    s.close()
    events = _cat_events(tmp_path / "open.jsonl")
    assert [e["name"] for e in events] == ["cat_contact_start", "cat_contact_end"]
    end = events[1]["evaluation_only_truth"]
    assert end["end_reason"] == "close" and end["peak_force_n"] > 0.0 and end["impulse_ns"] > 0.0
    assert end["duration_s"] > 0.04


def test_a_touch_finished_at_episode_end_is_not_counted_again():
    """The window keeps physics running after the episode ends; the same uninterrupted touch
    must not become a second contact."""
    s = RobotSystem(cats=1, cat_seed=0)
    s.reset(-2.0, 0.0, 0.0, (4.0, 4.0))
    _inject(s, [(-1.2, 0.0, math.pi, 0.4)])
    while not s.cat_contacts:
        s.advance(C.PHYSICS_DT)
    s.advance(0.02)
    s.finish_cat_contacts("episode_end")
    before = (s.cat_contacts, s.collisions, len(s.cat_contact_events))
    s.advance(3 * C.COLLISION_EVENT_GAP)
    assert (s.cat_contacts, s.collisions, len(s.cat_contact_events)) == before and not s._cat_open
    assert s.cats.cats[0].frozen
    s.close()


def test_a_cat_finished_while_touching_stays_frozen_until_a_full_gap():
    """After an early finish, a separation shorter than the collision gap keeps the cat frozen
    and recontact is not counted; only a continuous full gap unfreezes it, and a later touch
    is a new contact and collision."""
    s = RobotSystem(cats=1, cat_seed=0)
    s.reset(-2.0, 0.0, 0.0, (4.0, 4.0))
    _inject(s, [(1.0, 0.0, 0.0, 0.0)])  # far away; contacts are scripted below
    cat = s.cats.cats[0]

    def touch(on, seconds):
        s.sim.cat_contacts_now = (lambda: {0: []}) if on else (lambda: {})
        s.advance(seconds)

    touch(True, 0.1)
    s.finish_cat_contacts("episode_end")
    assert (s.cat_contacts, s.collisions) == (1, 1)
    touch(False, 0.05)  # shorter than the gap
    assert cat.frozen
    touch(True, 0.05)  # the same touch resumes: not counted
    assert cat.frozen and (s.cat_contacts, s.collisions) == (1, 1)
    touch(False, 0.05)
    assert cat.frozen  # the gap restarted at the last touch
    touch(False, C.COLLISION_EVENT_GAP)
    assert not cat.frozen
    touch(True, 0.05)
    assert (s.cat_contacts, s.collisions) == (2, 2) and cat.frozen
    s.close()


def test_reset_with_an_open_contact_finishes_a_valid_log(tmp_path):
    """One log per episode: reset finishes the open contact (end_reason reset), closes the old
    log so it validates, and detaches it."""
    from robot_env.drive_log import DriveLog, read_log
    s = RobotSystem(cats=1, cat_seed=0)
    s.log = DriveLog(tmp_path / "reset.jsonl")
    s.reset(-2.0, 0.0, 0.0, (4.0, 4.0))
    _inject(s, [(-1.2, 0.0, math.pi, 0.4)])
    while not s.cat_contacts:
        s.advance(C.PHYSICS_DT)
    s.advance(0.02)
    s.reset(-2.0, 0.0, 0.0, (4.0, 4.0))
    assert s.log is None and s.cat_contacts == 0
    read_log(tmp_path / "reset.jsonl")  # validates
    ends = [e for e in _cat_events(tmp_path / "reset.jsonl") if e["name"] == "cat_contact_end"]
    assert len(ends) == 1 and ends[0]["evaluation_only_truth"]["end_reason"] == "reset"
    s.close()


def test_gym_episode_end_finishes_the_cat_contact_in_the_log(tmp_path):
    from robot_env.drive_log import DriveLog
    from robot_env.env import RobotGoalEnv
    env = RobotGoalEnv(cats=1, cat_seed=0)
    env.system.log = DriveLog(tmp_path / "gym.jsonl")
    env.reset(seed=1000)
    s = env.system
    x, y, yaw = s.sim.true_pose()
    _inject(s, [(x + 0.8 * math.cos(yaw), y + 0.8 * math.sin(yaw), yaw + math.pi, 0.4)])
    for _ in range(100):
        _, _, terminated, truncated, _ = env.step(np.zeros(2, dtype=np.float32))
        if terminated or truncated:
            break
    assert terminated
    env.close()
    end = _cat_events(tmp_path / "gym.jsonl")[-1]
    assert end["name"] == "cat_contact_end" and end["evaluation_only_truth"]["end_reason"] == "episode_end"


def test_a_brief_separation_is_one_contact_with_one_start_and_one_end(tmp_path):
    """Touch, separate for less than the collision gap, touch again, then separate for good:
    one contact, one start, one end, and the log validates."""
    from robot_env.drive_log import DriveLog
    s = RobotSystem(cats=1, cat_seed=0)
    s.log = DriveLog(tmp_path / "flicker.jsonl")
    s.reset(-2.0, 0.0, 0.0, (4.0, 4.0))
    _inject(s, [(1.0, 0.0, 0.0, 0.0)])  # far away; contacts are scripted below
    plan = [(0.1, True), (0.05, False), (0.1, True), (0.5, False)]
    for seconds, touching in plan:
        s.sim.cat_contacts_now = (lambda: {0: []}) if touching else (lambda: {})
        s.advance(seconds)
    assert s.cat_contacts == 1 and not s._cat_open
    s.close()
    events = _cat_events(tmp_path / "flicker.jsonl")
    assert [e["name"] for e in events] == ["cat_contact_start", "cat_contact_end"]
    end = events[1]["evaluation_only_truth"]
    assert end["end_reason"] == "separated" and abs(end["duration_s"] - 0.25) < 0.01


def test_cat_truth_never_leaks_outside_evaluation_only_truth(tmp_path):
    """Every record type: cat identity, state, initiator, and forces appear only under
    evaluation_only_truth."""
    from robot_env.drive_log import DriveLog, read_log
    s = RobotSystem(cats=2, cat_seed=0)
    s.log = DriveLog(tmp_path / "leak.jsonl")
    s.reset(-2.0, 0.0, 0.0, (4.0, 4.0))
    _inject(s, [(-1.2, 0.0, math.pi, 0.4), (1.0, 0.0, 0.0, 0.0)])
    s.cats._enter(s.cats.cats[1], "sit")
    s.advance(3.0)
    s.close()
    secret = {"cat", "cats", "state", "initiator", "peak_force_n", "impulse_ns", "robot_closing_speed",
              "cat_closing_speed", "duration_s", "end_reason", "cat_contacts"}
    names = set()

    def check(obj, path):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k == "evaluation_only_truth":
                    continue
                assert k not in secret or path == ("header",), (path, k)
                check(v, path + (k,))

    for r in read_log(tmp_path / "leak.jsonl"):
        names.add(r.get("name"))
        check(r, (r["kind"],))
    assert {"cat_state", "cat_contact_start", "cat_contact_end"} <= names


def test_gym_episode_ends_on_cat_contact():
    from robot_env.env import RobotGoalEnv
    env = RobotGoalEnv(cats=1, cat_seed=0)
    env.reset(seed=1000)
    s = env.system
    x, y, yaw = s.sim.true_pose()
    _inject(s, [(x + 0.8 * math.cos(yaw), y + 0.8 * math.sin(yaw), yaw + math.pi, 0.4)])
    terminated = truncated = False
    for _ in range(100):
        _, _, terminated, truncated, _ = env.step(np.zeros(2, dtype=np.float32))
        if terminated or truncated:
            break
    assert terminated and s.cat_contacts == 1
    env.close()


def test_gym_rejects_cat_arguments_that_disagree_with_a_given_system():
    from robot_env.env import RobotGoalEnv
    s = RobotSystem(cats=1, cat_seed=2)
    with pytest.raises(ValueError):
        RobotGoalEnv(system=s, cats=2, cat_seed=2)
    with pytest.raises(ValueError):
        RobotGoalEnv(system=s, cats=1, cat_seed=3)
    RobotGoalEnv(system=s, cats=1, cat_seed=2).close()
    with pytest.raises(ValueError):
        RobotGoalEnv(system=RobotSystem(), cats=1)


def test_lidar_with_one_cat_partly_hiding_another():
    """A near cat partly hides a far one: rays on the near cat read the near distance, and the
    far cat is still seen past the near cat's edge."""
    s = RobotSystem(cats=2, cat_seed=0)
    s.reset(-2.0, 0.0, 0.0, (4.0, 4.0))
    near, far = s.cats.cats
    near.x, near.y, near.yaw = -1.2, 0.0, 0.0  # end-on, 0.15 m wide, about 0.7 m away
    far.x, far.y, far.yaw = 0.4, 0.3, 0.0  # 2.4 m away, offset so only part of it is hidden
    s.cats._write_poses()
    mujoco_forward(s.sim)
    ranges, valid = s.sim.scan()
    m = s.sim.model
    body = np.array([m.body(int(m.geom_bodyid[g])).name if g >= 0 else "" for g in s.sim._scan_geom])
    on_near, on_far = body == "cat0", body == "cat1"
    assert on_near.any() and on_far.any()
    assert (ranges[on_near & valid] < 1.0).all() and (ranges[on_far & valid] > 2.0).all()
    # partly hidden: the far cat's full angular span is wider than the part the lidar sees
    span = np.degrees(np.arctan2(far.y + np.array([-0.075, 0.075]), far.x + 2.0))
    in_span = (np.degrees(s.sim.lidar_angles) > span[0]) & (np.degrees(s.sim.lidar_angles) < span[1])
    assert (in_span & on_near).any() and (in_span & on_far).any()
    s.close()


def test_sit_and_dart_states_have_their_durations_and_are_logged(tmp_path):
    """Forced states: sit holds still for 3 to 8 s with the body pitched; dart runs fast for 0.4
    to 0.8 s. Every transition reaches the drive log with its time."""
    from robot_env.cats import SIT_PITCH
    from robot_env.drive_log import DriveLog, read_log
    s = RobotSystem(cats=1, cat_seed=7)
    with DriveLog(tmp_path / "states.jsonl") as log:
        s.log = log
        s.reset(-2.0, 0.0, 0.0, (4.0, 4.0))
        herd, cat = s.cats, s.cats.cats[0]
        cat.x, cat.y, cat.v, cat.w = 2.0, 0.0, 0.0, 0.0  # open corridor, far from the robot
        herd._enter(cat, "sit")
        assert 3.0 <= cat.state_until - herd.time <= 8.0
        x0, y0 = cat.x, cat.y
        s.advance(1.0)
        assert cat.state == "sit" and (cat.x, cat.y) == (x0, y0)
        assert abs(cat.pitch - SIT_PITCH) < 1e-6
        herd._enter(cat, "dart")
        dart_len = cat.state_until - herd.time
        assert 0.4 <= dart_len <= 0.8
        start, top = herd.time, 0.0
        while herd.time < start + dart_len - 0.02:
            s.advance(C.CONTROL_PERIOD)
            top = max(top, cat.v)
        assert cat.state == "dart" and top > 0.4  # faster than any walk (0.15 to 0.35 m/s)
        s.log = None
    events = [r for r in read_log(tmp_path / "states.jsonl") if r["kind"] == "event" and r["name"] == "cat_state"]
    assert {"sit", "dart"} <= {e["evaluation_only_truth"]["state"] for e in events}
    assert all(e["evaluation_only_truth"]["cat"] == 0 and e["t"] >= 0.0 for e in events)
    s.close()


def test_drive_log_header_is_bound_to_the_system(tmp_path):
    """The header's cats and seed come from the system; a caller value that disagrees fails the log."""
    from robot_env.drive_log import DriveLog, read_log
    s = RobotSystem(cats=2, cat_seed=5)
    with DriveLog(tmp_path / "auto.jsonl") as log:
        s.log = log
        s.reset(0.0, 0.0, 0.0, (4.0, 4.0))
        s.advance(C.CONTROL_PERIOD)
        s.log = None
    header = read_log(tmp_path / "auto.jsonl")[0]
    assert header["cats"] == 2 and header["cat_seed"] == 5
    bad = DriveLog(tmp_path / "bad.jsonl", cats=1, cat_seed=5)
    s.log = bad
    s.reset(0.0, 0.0, 0.0, (4.0, 4.0))
    s.log = None
    assert bad.failed and "cats" in bad.error
    bad.close()
    s.close()


def test_drive_log_records_cats_under_evaluation_truth(tmp_path):
    from robot_env.drive_log import DriveLog, read_log
    s = RobotSystem(cats=2, cat_seed=3)
    with DriveLog(tmp_path / "cats.jsonl", cats=2, cat_seed=3) as log:
        s.log = log
        s.reset(0.0, 0.0, 0.0, (4.0, 4.0))
        for _ in range(10):
            s.advance(C.CONTROL_PERIOD)
    s.log = None
    recs = read_log(tmp_path / "cats.jsonl")
    assert recs[0]["cats"] == 2 and recs[0]["cat_seed"] == 3
    tick = [r for r in recs if r["kind"] == "tick"][-1]
    truth = tick["evaluation_only_truth"]
    assert len(truth["cats"]) == 2 and {"pose", "v", "w", "pitch", "state"} <= set(truth["cats"][0])
    assert "cats" not in tick or tick.get("cats") is None  # cats only under evaluation truth
    s.close()


def test_baseline_driver_with_cats_reports_every_outcome():
    """Demo evidence only (one seeded run): the baseline reaches the goal and every contact,
    intervention, and timeout is reported."""
    from robot_env.baseline import BaselineDriver
    from robot_env.layout import RoomMap
    s = RobotSystem(cats=2, cat_seed=1)
    task = RoomMap(s.sim.model).sample_task(1000)
    s.reset(*task.start, task.goal)
    d = BaselineDriver()
    while s.time < 60.0 and s.observe().goal_distance > C.GOAL_RADIUS and not s.collisions:
        s.apply(d.decide(s.observe()))
        s.advance(C.DECISION_PERIOD)
    assert s.collisions == 0 and s.cat_contacts == 0, s.cat_contact_events
    assert s.observe().goal_distance <= C.GOAL_RADIUS
    s.close()
