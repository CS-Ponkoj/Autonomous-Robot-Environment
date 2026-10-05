"""Wandering cats: geometry, motion bounds, sensing, contacts, determinism."""

import math

import mujoco
import numpy as np
import pytest

from robot_env import config as C
from robot_env.cat_motion import SIT_UP_TIME
from robot_env.cats import (ACCEL, ANIM_PERIOD, CAT_GAP, MAX_CATS, MIN_TURN_SPEED, PIVOT_RATE, ROBOT_GAP, V_MAX, WALL_GAP,
                            YAW_ACCEL, YAW_RATE, cat_world)
from robot_env.sim import RobotSim
from robot_env.system import RobotSystem


def test_cats_off_keeps_the_default_world_unchanged():
    """With no cats the system loads the plain world (no cat bodies at all)."""
    plain = RobotSim()
    s = RobotSystem()
    assert s.cats is None and s.sim.model.nbody == plain.model.nbody and s.sim.model.ngeom == plain.model.ngeom
    assert not any(s.sim.model.body(i).name.startswith("cat") for i in range(s.sim.model.nbody))
    s.close()
    plain.close()


def test_cat_count_and_seed_are_validated():
    assert cat_world(MAX_CATS).worldbody and cat_world(0).worldbody == ""
    with pytest.raises(ValueError):
        cat_world(MAX_CATS + 1)
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
    """A long seeded run. At every motion sample (100 Hz) the whole visible cat keeps its gaps and
    the root speed, acceleration, turn rate, and turn acceleration stay inside their bounds; a
    cat never turns while (nearly) stopped. At every physics step every collider moves
    continuously (no jumps). No solver warnings."""
    prev, prev_cols = {}, None
    worst = {"accel": 0.0, "yaw_rate": 0.0, "yaw_accel": 0.0, "speed": 0.0, "col_step": 0.0}

    def check(s, i):
        nonlocal prev_cols
        herd = s.cats
        cols = herd.col_world.copy()
        if prev_cols is not None:
            worst["col_step"] = max(worst["col_step"], float(np.abs(cols - prev_cols).max()))
        prev_cols = cols
        for cat in herd.cats:
            key = (cat.index, cat.t0)
            if key in prev:
                continue
            prev[key] = True
            wall, robot, other = herd.gaps(cat, cat.x, cat.y, cat.yaw)
            assert wall >= WALL_GAP - 1e-9 and robot >= ROBOT_GAP - 1e-9 and other >= CAT_GAP - 1e-9, (i, cat.index)
            assert abs(cat.x) < C.FLOOR_HALF_SIZE and abs(cat.y) < C.FLOOR_HALF_SIZE
            last = prev.get(("state", cat.index))
            if last is not None:
                pv, pw, pyaw = last
                worst["accel"] = max(worst["accel"], abs(cat.v - pv) / ANIM_PERIOD)
                worst["yaw_accel"] = max(worst["yaw_accel"], abs(cat.w - pw) / ANIM_PERIOD)
                if max(abs(pv), abs(cat.v)) < MIN_TURN_SPEED - 1e-9:  # (nearly) stopped: pivot steps only
                    assert abs(math.remainder(cat.yaw - pyaw, 2 * math.pi)) <= PIVOT_RATE * ANIM_PERIOD + 1e-9, (i, cat.index)
            prev[("state", cat.index)] = (cat.v, cat.w, cat.yaw)
            worst["speed"] = max(worst["speed"], abs(cat.v))
            worst["yaw_rate"] = max(worst["yaw_rate"], abs(cat.w))

    s = _run(each=check)
    assert worst["speed"] <= V_MAX + 1e-9 and worst["yaw_rate"] <= YAW_RATE + 1e-9
    assert worst["accel"] <= ACCEL + 1e-6 and worst["yaw_accel"] <= YAW_ACCEL + 1e-6
    assert worst["col_step"] <= 0.003, worst  # continuous colliders: at most 3 mm per 0.5 ms step
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
        cat_geoms = np.flatnonzero(s.sim._cat_geom)
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


def test_lidar_sees_a_standing_cat_from_every_side():
    """At 0.4 to 2.0 m, every yaw (45 deg steps) and bearing offset between rays (0 to 0.75 deg),
    at least one ray's first hit is the cat at about its distance; at 3.0 m at least 95% of the
    cases. A thin leg can fall between rays farther away (real lidar behaviour)."""
    s = RobotSystem(cats=1, cat_seed=0)
    s.reset(-4.0, 0.0, 0.0, (4.0, 4.0))  # corridor, robot facing +x
    misses, total_far, misses_far = [], 0, 0
    for dist in (0.4, 0.8, 1.2, 2.0, 3.0):
        for offset_deg in (0.0, 0.25, 0.5, 0.75):
            for k in range(8):
                bearing = math.radians(offset_deg)
                s.cats.place(0, -4.0 + dist * math.cos(bearing), dist * math.sin(bearing), k * math.pi / 4, state="pause")
                ranges, valid = s.sim.scan()
                near = np.abs(np.degrees(s.sim.lidar_angles) - offset_deg) <= 25.0
                seen = (near & valid & _cat_rays(s) & (ranges > dist - 0.35) & (ranges < dist + 0.3)).any()
                if dist >= 3.0:
                    total_far += 1
                    misses_far += not seen
                elif not seen:
                    misses.append((dist, offset_deg, k))
    assert not misses, misses
    assert misses_far <= 0.05 * total_far, (misses_far, total_far)
    s.close()


def _cat_rays(s):
    m = s.sim.model
    return np.array([g >= 0 and m.body(int(m.geom_bodyid[g])).name.startswith("cat") for g in s.sim._scan_geom])


def _inject(s, poses):
    """Fault injection (the cat policy never does this): place cats and drive them straight,
    bypassing the policy's gaps and behavior."""
    s.cats.pose_ok = lambda *a, **k: True
    s.cats.tick = lambda **k: None

    def straight(cat):  # no local planner either: straight ahead at the placed speed
        cat.cmd_v, cat.cmd_w, cat.cmd_lat = cat.target_v, 0.0, 0.0
    s.cats._plan = straight
    for i, (x, y, yaw, v) in enumerate(poses):
        s.cats.place(i, x, y, yaw, v=v, state="walk")


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
    assert cat.x - s.sim.true_pose()[0] > 0.15  # no tunneling: the cat stays in front of the robot
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
    """A near cat partly hides a far one: rays on the near cat read the near distance, the far cat
    is still seen past the near cat's edge, and fewer rays reach it than without the near cat."""
    s = RobotSystem(cats=2, cat_seed=0)
    s.reset(-4.0, 0.0, 0.0, (4.0, 4.0))
    m = s.sim.model

    def owners():
        ranges, valid = s.sim.scan()
        body = np.array([m.body(int(m.geom_bodyid[g])).name if g >= 0 else "" for g in s.sim._scan_geom]).astype(str)
        return ranges, valid, np.char.startswith(body, "cat0_"), np.char.startswith(body, "cat1_")

    s.cats.place(0, 3.0, -3.0, 0.0, state="pause")  # out of the way
    s.cats.place(1, -1.3, 0.35, math.pi / 2, state="pause")  # 2.7 m away
    alone = owners()[3].sum()
    s.cats.place(0, -3.0, 0.0, math.pi / 2, state="pause")  # broadside, about 1 m away
    ranges, valid, on_near, on_far = owners()
    assert on_near.any() and on_far.any() and on_far.sum() < alone, (on_far.sum(), alone)
    assert (ranges[on_near & valid] < 1.4).all() and (ranges[on_far & valid] > 2.2).all()
    s.close()


def test_sit_and_dart_states_have_their_durations_and_are_logged(tmp_path):
    """Forced states: sit holds still for 3 to 8 s (the cat sits down); dart runs fast for 0.4 to
    0.8 s once the cat has got up. Every transition reaches the drive log with its time."""
    from robot_env.drive_log import DriveLog, read_log
    s = RobotSystem(cats=1, cat_seed=7)
    with DriveLog(tmp_path / "states.jsonl") as log:
        s.log = log
        s.reset(-2.0, 0.0, 0.0, (4.0, 4.0))
        herd, cat = s.cats, s.cats.cats[0]
        herd.place(0, 2.0, 0.0, 0.0)  # open corridor, far from the robot
        herd._enter(cat, "sit")
        assert 3.0 <= cat.state_until - herd.time <= 8.0
        x0, y0 = cat.x, cat.y
        s.advance(1.0)
        assert cat.state == "sit" and math.hypot(cat.x - x0, cat.y - y0) < 0.02  # settles where it is
        assert cat.animator.sit == 1.0  # sat down
        up = cat.animator.sit * SIT_UP_TIME  # getting up comes first
        herd._enter(cat, "dart")
        dart_len = cat.state_until - herd.time - up
        assert 0.4 <= dart_len <= 0.8
        while cat.animator.sit > 0.0:
            s.advance(C.CONTROL_PERIOD)
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


def test_a_cat_sits_only_when_still_and_is_up_before_it_moves():
    """Sitting: a walking cat told to sit first stops (no sitting while it moves), then sits down
    with its paws planted where they are (none steps while sitting) and its root fixed; told to
    flee, it gets up first (its root held until it is up), then moves, every gap kept."""
    from robot_env.cat_motion import SIT_DOWN_TIME
    from robot_env.cats import SIT_MOVE
    s = RobotSystem(cats=1, cat_seed=7)
    s.reset(-4.0, -2.5, 0.0, (4.0, 2.5))
    herd, cat = s.cats, s.cats.cats[0]
    herd.place(0, 2.0, 0.0, 0.0, v=0.25, state="walk")
    s.advance(0.5)
    herd._enter(cat, "sit")
    while abs(cat.v) >= 0.01:
        assert cat.animator.sit == 0.0  # not while it is still moving
        s.advance(C.CONTROL_PERIOD)
    while cat.animator.sit == 0.0:  # paws settling under the body
        s.advance(C.CONTROL_PERIOD)
    x0, y0, yaw0 = cat.x, cat.y, cat.yaw
    paws = {n: leg.world.copy() for n, leg in cat.animator.legs.items()}
    s.advance(SIT_DOWN_TIME + 0.2)
    assert cat.animator.sit == 1.0 and cat.state == "sit"
    assert (cat.x, cat.y, cat.yaw) == (x0, y0, yaw0)
    for n, leg in cat.animator.legs.items():
        assert leg.planted and np.allclose(leg.world, paws[n])
    herd._enter(cat, "flee")
    cat.target_v = 0.35
    while cat.animator.sit > SIT_MOVE:
        assert (cat.x, cat.y) == (x0, y0)  # getting up: not moving yet
        s.advance(C.CONTROL_PERIOD)
    s.advance(1.5)
    assert math.hypot(cat.x - x0, cat.y - y0) > 0.1  # then off
    g = herd.current_gaps()[0]
    assert g["wall"] >= WALL_GAP - 0.001 and s.cat_contacts == 0
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


# ----- doorways: cats pass through them, never rest there, and give way to a waiting robot -----
def test_cats_never_rest_in_a_doorway_or_the_corridor():
    s = RobotSystem(cats=3, cat_seed=8)
    s.reset(-4.0, 4.0, 0.0, (4.0, -4.0))  # robot parked in the office corner, out of the way
    herd = s.cats
    rests = 0
    for _ in range(int(120.0 / C.CONTROL_PERIOD)):
        s.advance(C.CONTROL_PERIOD)
        for cat in herd.cats:
            if cat.state in ("sit", "pause"):
                rests += 1
                assert not herd.in_choke(cat), (cat.index, cat.state, round(cat.x, 2), round(cat.y, 2))
    assert rests > 0  # the cats did rest (elsewhere)
    s.close()


def _doorway_robot(door, cat_state="pause"):
    """One cat resting in a doorway of the corridor's north wall (facing into the room) and the
    robot in the corridor 1.0 m away, facing the doorway."""
    dx, dy = door
    s = RobotSystem(cats=1, cat_seed=2)
    s.reset(dx, dy - 1.0, math.pi / 2, (dx, 3.0))
    s.cats.place(0, dx, dy + 0.1, math.pi / 2, state=cat_state)
    cat = s.cats.cats[0]
    wall, robot, _ = s.cats.gaps(cat, cat.x, cat.y, cat.yaw)
    assert wall >= WALL_GAP and robot >= ROBOT_GAP  # a pose the cat itself could be in
    return s


@pytest.mark.parametrize("door", [(-2.5, 0.75), (2.5, 0.75)])
def test_a_cat_in_a_doorway_gives_way_to_a_waiting_robot(door):
    """The robot asks to drive through; its safety layer holds it back because the cat is in the
    way; the cat starts moving aside within 0.5 s and the robot is through the doorway within
    8 s, with no contact (the robot never pushes a cat)."""
    s = _doorway_robot(door)
    cat = s.cats.cats[0]
    waited_from = yielded_at = through_at = None
    for k in range(int(8.0 / C.CONTROL_PERIOD)):
        if k % 5 == 0:
            s.drive(0.3, 0.0)  # a fresh forward request every 0.1 s
        s.advance(C.CONTROL_PERIOD)
        if waited_from is None and "clearance" in s.last_result.reasons and s.last_result.command.v < 0.28:
            waited_from = s.time
        if yielded_at is None and cat.state == "yield":
            yielded_at = s.time
        if s.sim.true_pose()[1] > door[1] + 0.4:
            through_at = s.time
            break
    # it gives way as the robot comes, or at the latest 0.5 s after the robot starts waiting
    assert yielded_at is not None
    assert waited_from is None or yielded_at <= waited_from + 0.5 + C.CONTROL_PERIOD
    assert through_at is not None and through_at <= 8.0
    assert s.collisions == 0 and s.cat_contacts == 0
    s.close()


def test_a_parked_robot_does_not_herd_a_cat_out_of_its_way():
    s = _doorway_robot((2.5, 0.75))
    cat = s.cats.cats[0]
    for _ in range(int(3.0 / C.CONTROL_PERIOD)):
        s.advance(C.CONTROL_PERIOD)
        assert cat.state == "pause"
    s.close()


def test_the_skin_is_drawn_at_the_current_interpolated_pose():
    """Rendering half-way between two animation samples: every bone body is where the cats just
    put it (the kinematics are propagated after the bones are written), for the window views
    and the robot camera alike."""
    s = RobotSystem(cats=2, cat_seed=3)
    s.reset(-4.0, 0.0, 0.0, (4.0, 4.0))
    s.cats.place(0, -2.5, 0.0, 0.0, v=0.4, state="walk")
    s.advance(1.0 + ANIM_PERIOD / 2)  # mid-sample
    m, d = s.sim.model, s.sim.data
    s.sim.before_render()
    ids = [m.body(f"cat0_b{j}").id for j in range(len(s.cats.rig.joint_names))]
    mocap = [m.body_mocapid[i] for i in ids]
    assert np.abs(d.xpos[ids] - d.mocap_pos[mocap]).max() < 1e-9
    s.sim.render_camera()
    assert np.abs(d.xpos[ids] - d.mocap_pos[mocap]).max() < 1e-9
    s.close()


def test_a_cat_behind_a_wall_does_not_react_to_the_robot():
    """The robot pushes toward the corridor's solid north wall (a fresh forward request held back
    by safety: it is waiting) with a cat resting just behind that wall, close enough to be scared
    if the wall were not there. The cat does not see it: no fleeing, no giving way, no watching."""
    s = RobotSystem(cats=1, cat_seed=2)
    s.reset(1.0, 0.45, math.pi / 2, (1.0, 3.0))
    s.cats.place(0, 1.0, 1.0, 0.0, state="pause")
    cat = s.cats.cats[0]
    for k in range(int(3.0 / C.CONTROL_PERIOD)):
        if k % 5 == 0:
            s.drive(0.3, 0.0)
        s.advance(C.CONTROL_PERIOD)
        assert cat.state == "pause" and not cat.sees_robot
        assert cat.animator.head_yaw == pytest.approx(0.0, abs=1e-9)
    assert s._waiting_forward()  # the robot really was waiting the whole time
    s.close()


def test_the_same_cat_in_plain_sight_does_react():
    """Control for the test above: the robot in the open office at the same distance, waiting
    with the cat in its path, so the cat gives way."""
    s = RobotSystem(cats=1, cat_seed=2)
    s.reset(-3.0, 1.7, math.pi / 2, (-3.0, 4.0))
    s.cats.place(0, -3.0, 2.3, 0.0, state="pause")
    cat = s.cats.cats[0]
    reacted = False
    for k in range(int(3.0 / C.CONTROL_PERIOD)):
        if k % 5 == 0:
            s.drive(0.3, 0.0)
        s.advance(C.CONTROL_PERIOD)
        reacted |= cat.state in ("yield", "flee")
    assert cat.sees_robot or reacted
    assert reacted
    s.close()


@pytest.mark.parametrize("door,side,level", [(1, -1, 1), (2, 1, 4), (4, 1, 4), (3, -1, 0), (1, -1, 3), (1, -1, 4)])
def test_doorway_matrix_samples(door, side, level):
    """A sample of tools/doorway_check.py (all 100 trials are the release gate): the cat gives way
    in time, the robot is through in time, nothing touches, and the cat as drawn keeps the robot's
    safety buffer from the robot's footprint at every physics step (including the fast levels
    through the lab door, where a cat yielding at the jamb once came within 2 mm)."""
    import importlib
    gate = importlib.import_module("tools.doorway_check")
    trial = next(t for t in gate.trials() if t[0] == door and t[4] == side and t[5] == level and t[6] == "pause")
    r = gate.run_trial(trial)
    v = C.SPEED_LEVELS[level][0]
    assert r["placed"] and r["contacts"] == 0 and r["collisions"] == 0 and r["yield"] is not None
    assert r["low"] >= gate.MIN_FOOTPRINT_GAP
    if r["wait"] is not None:
        assert r["yield"] <= r["wait"] + gate.YIELD_AFTER_WAIT + 0.02
        assert r["clear"] is not None and r["clear"] <= r["wait"] + gate.CLEAR_AFTER + 0.02
    assert r["through"] is not None and r["through"] <= (r["start"] + 0.4) / v + gate.WAIT_ALLOWANCE


def test_a_cat_sees_past_a_door_jamb_with_its_eyes():
    """Sight is cast from the eyes (either eye), not the body: a cat with its head past the lab
    door's jamb sees the robot in the corridor although a ray from its body would hit the jamb.
    The eye positions are those of the pose as drawn."""
    s = RobotSystem(cats=1, cat_seed=2)
    s.reset(1.9, 0.0, math.pi / 2, (2.5, 3.0))
    s.cats.place(0, 3.25, 1.0, math.pi, state="pause")
    herd, cat = s.cats, s.cats.cats[0]
    assert not s.sim.sees_robot((cat.x, cat.y, 0.22))  # the body is behind the jamb
    s.advance(C.CONTROL_PERIOD)
    assert cat.sees_robot
    s.sim.before_render()
    m, d = s.sim.model, s.sim.data
    names = herd.rig.joint_names
    drawn = [d.xpos[m.body(f"cat0_b{names.index(n)}").id] for n in ("Character1_EyeL_00", "Character1_EyeR_025")]
    assert np.abs(herd.eyes(cat) - np.array(drawn)).max() < 1e-6
    s.close()


# ----- the per-sample proof: swept moves, inside a margin, and the fallbacks -----
def _herd_one(x, y, yaw, cats=1):
    s = RobotSystem(cats=cats, cat_seed=2)
    s.reset(-4.3, 4.3, 0.0, (4.0, -4.0))  # robot parked in the office corner
    s.cats.place(0, x, y, yaw, state="pause")
    return s, s.cats, s.cats.cats[0]


def test_a_move_whose_ends_are_clear_but_passes_too_close_is_refused():
    """Past the corner of the storage boxes: both ends of a 12 cm move keep the 3 cm wall gap, but
    the straight path between them cuts to about 1.7 cm from the corner. The swept proof refuses
    it (an end-point check would not); the same move in open space is accepted."""
    from robot_env.cats import Sample
    s, herd, cat = _herd_one(0.3, -3.9, 0.0)
    start = herd._finish(Sample(0.4, -3.6, 2.7489, cat.next.pos, cat.next.rot))
    end = (0.46, -3.7039, 2.7489)
    path = [herd.gaps(cat, 0.4 + (end[0] - 0.4) * f, -3.6 + (end[1] + 3.6) * f, end[2], start)[0]
            for f in np.linspace(0.0, 1.0, 25)]
    assert path[0] >= WALL_GAP and path[-1] >= WALL_GAP and min(path) < WALL_GAP - 0.01  # a real dip
    assert herd.pose_ok(cat, *end, pose=start)  # an end-point check alone would allow it
    assert not herd.pose_ok(cat, *end, pose=start, sweep_from=start)
    s.close()
    s, herd, cat = _herd_one(-2.5, 2.4, 0.0)
    start = herd._finish(Sample(-2.5, 2.4, 2.7489, cat.next.pos, cat.next.rot))
    assert herd.pose_ok(cat, -2.44, 2.2961, 2.7489, pose=start, sweep_from=start)
    s.close()


def test_another_cat_moving_in_the_same_sample_counts_by_its_path():
    """Cat 1 already moved this sample straight across the space in front of cat 0: cat 0 standing
    still is clear of where cat 1 ended, but not of the path it took."""
    from robot_env.cats import ANIM_PERIOD as P, Sample
    s, herd, a = _herd_one(-2.5, 2.6, 0.0, cats=2)
    b = herd.cats[1]
    herd.place(1, -1.9, 2.0, math.pi / 2, state="pause")
    b.prev = herd._finish(Sample(-1.9, 3.2, math.pi / 2, b.next.pos, b.next.rot))  # it came from beyond cat 0
    b.t0 = herd.time  # ... in this very sample
    assert herd.pose_ok(a, a.x, a.y, a.yaw)  # clear of where cat 1 is now
    assert not herd.pose_ok(a, a.x, a.y, a.yaw, sweep_from=a.next)  # not of its path
    b.prev = b.next  # cat 1 did not move: fine
    assert herd.pose_ok(a, a.x, a.y, a.yaw, sweep_from=a.next)
    s.close()


def test_inside_a_gap_a_cat_may_only_move_away():
    """A cat whose own gap is already too small (here: placed 2 cm from the storage boxes, inside
    the 3 cm wall gap) may move away from the boxes, but not any closer."""
    from robot_env.cats import Sample
    s, herd, cat = _herd_one(0.3, -3.9, 0.0)
    start = herd._finish(Sample(0.43, -3.652, 2.7489, cat.next.pos, cat.next.rot))
    g = herd.gaps(cat, start.x, start.y, start.yaw, start)[0]
    assert g < WALL_GAP
    # a 1 mm step (about what one 10 ms sample moves when edging out) straight away from the boxes
    steps = [(0.001 * math.cos(a), 0.001 * math.sin(a)) for a in np.linspace(0, 2 * math.pi, 16, endpoint=False)]
    dx, dy = max(steps, key=lambda d: herd.gaps(cat, start.x + d[0], start.y + d[1], start.yaw, start)[0])
    away = herd.pose_ok(cat, start.x + dx, start.y + dy, start.yaw, pose=start, sweep_from=start)
    closer = herd.pose_ok(cat, start.x - dx, start.y - dy, start.yaw, pose=start, sweep_from=start)
    g_away = herd.gaps(cat, start.x + dx, start.y + dy, start.yaw, start)[0]
    assert g_away > g and away and not closer
    s.close()


@pytest.mark.parametrize("fails", [1, 2, 10])
def test_animation_fallbacks(fails):
    """The animated pose refused once: a calm animation is used (the cat still moves; its paws do
    exactly what a calm step does); twice: it stands as it is (root and posture held, the command
    refused, a new plan at once); every time (a paw in the air cannot be put down either): the
    whole sample is rejected (old root and pose, the command refused, counted). Standing or
    rejected, every planted paw stays planted where it is."""
    import copy

    from robot_env.cats import ANIM_PERIOD
    s, herd, cat = _herd_one(-2.5, 2.4, 0.0)
    cat.state, cat.target_v, cat.target_yaw = "walk", 0.3, 0.0
    s.advance(0.5)  # walking
    real = herd.pose_ok
    left = {"n": fails}

    def flaky(c, x, y, yaw, margin=0.0, pose=None, sweep_from=None):
        if pose is not None and c is cat and left["n"] > 0:
            left["n"] -= 1
            return False
        return real(c, x, y, yaw, margin, pose, sweep_from)
    herd.pose_ok = flaky
    if fails == 10:
        herd._landing_ok = lambda c, pose: False  # a paw cannot be put down short either
    root0, held0, now = (cat.x, cat.y, cat.yaw), cat.held, herd.time
    calm = copy.deepcopy(cat.animator)  # what a calm step from here does
    planted0 = {n: leg.world.copy() for n, leg in cat.animator.legs.items() if leg.planted}
    assert planted0
    herd._advance_sample(cat)
    herd.pose_ok = real
    command = (cat.cmd_v, cat.cmd_w, cat.cmd_lat)
    if fails == 1:
        assert cat.x > root0[0] and cat.held == held0 and command not in cat.rejected
        calm.tail_target = 0.0
        calm.update(ANIM_PERIOD, cat.x, cat.y, cat.yaw, cat.v, cat.w, None, cat.cmd_lat, settle=False)
        for n, leg in cat.animator.legs.items():
            assert leg.planted == calm.legs[n].planted and np.allclose(leg.world, calm.legs[n].world)
    else:
        assert (cat.x, cat.y, cat.yaw) == root0  # the root held exactly
        assert command in cat.rejected and cat.plan_at <= now + 1e-9  # refused; a new plan at once
        assert cat.held == held0 + (fails == 10)
        if fails == 10:
            assert cat.blocked and np.allclose(cat.next.pos, cat.prev.pos)  # the old pose
        for n, w in planted0.items():  # every planted paw stays planted where it is
            leg = cat.animator.legs[n]
            assert leg.planted and np.allclose(leg.world, w)
    for n, w in planted0.items():
        leg = cat.animator.legs[n]
        if leg.planted:
            assert np.allclose(leg.world, w)  # a planted paw never slides
    s.close()


@pytest.mark.parametrize("robot_v,robot_yaw,expect", [
    (0.7, 0.0, 0.91),  # coming straight at the cat: hurries (1.3 x 0.7 m/s)
    (-0.7, 0.0, 0.6),  # backing away: no hurry
    (0.7, math.pi / 2, 0.6),  # driving past (across the line to the cat): no hurry
    (0.0, 0.0, 0.6),  # stopped
    (1.0, 0.0, 1.0),  # fast: capped at the cat's top speed
])
def test_a_yielding_cat_hurries_by_how_fast_the_robot_closes_in(monkeypatch, robot_v, robot_yaw, expect):
    """The yield speed follows the robot's closing speed toward the cat (its velocity along the
    line to the cat), not its forward speed."""
    from robot_env.cats import V_MAX, yield_speed
    s, herd, cat = _herd_one(-1.0, 0.0, 0.0)
    monkeypatch.setattr(s.sim, "true_pose", lambda: (-2.0, 0.0, robot_yaw))
    monkeypatch.setattr(s.sim, "true_velocity", lambda: (robot_v, 0.0))
    closing = herd._robot_closing_speed(cat)
    assert yield_speed(closing) == pytest.approx(expect)
    assert yield_speed(closing) <= V_MAX
    s.close()


@pytest.mark.parametrize("robot_v,robot_yaw,expect", [(0.7, 0.0, 0.91), (-0.7, 0.0, 0.6), (0.7, math.pi / 2, 0.6)])
def test_a_yielding_cat_on_its_way_hurries_by_the_closing_speed(monkeypatch, robot_v, robot_yaw, expect):
    """At the call site: a cat already giving way (following its route to the spot) takes the
    speed from the robot's closing speed on each tick: hurried by a robot coming at it, not by
    one backing away or driving past."""
    s, herd, cat = _herd_one(-1.0, 0.0, 0.0)
    monkeypatch.setattr(s.sim, "true_pose", lambda: (-2.0, 0.0, robot_yaw))
    monkeypatch.setattr(s.sim, "true_velocity", lambda: (robot_v, 0.0))
    cat.state, cat.state_until, cat.target_v = "yield", herd.time + 5.0, 0.6
    cat.route, cat.destination = [(-1.0, 1.2), (-1.0, 1.6)], (-1.0, 1.6)
    herd.tick()
    assert cat.state == "yield" and cat.target_v == pytest.approx(expect)
    s.close()


def test_a_paw_put_down_short_lands_straight_below_where_it_is():
    """put_paws_down: a paw in the air lands at its present spot (only coming down), on the
    rest of its step, and then walks on normally."""
    from robot_env.cat_motion import CatAnimator
    from robot_env.cat_rig import load_rig
    from robot_env.cats import ANIM_PERIOD
    a = CatAnimator(load_rig(), 0)
    a.reset(0.0, 0.0, 0.0)
    x = 0.0
    while all(leg.planted for leg in a.legs.values()):
        x += 0.3 * ANIM_PERIOD
        a.update(ANIM_PERIOD, x, 0.0, 0.0, 0.3, 0.0)
    swinging = [leg for leg in a.legs.values() if not leg.planted]
    spots = {leg.name: leg.world[:2].copy() for leg in swinging}
    assert a.put_paws_down()
    while not all(leg.planted for leg in swinging):
        a.update(ANIM_PERIOD, x, 0.0, 0.0, 0.0, 0.0, settle=False, freeze=True)
        for leg in swinging:
            assert np.allclose(leg.world[:2], spots[leg.name])  # straight down
    for leg in swinging:
        assert leg.world[2] == pytest.approx(leg.neutral[2]) and not leg.short
    assert not a.put_paws_down()  # nothing in the air now


def _paw_in_the_air(herd, cat, s):
    cat.state, cat.target_v, cat.target_yaw = "walk", 0.3, cat.yaw
    for _ in range(400):  # walk until a paw is in the air
        s.advance(0.005)
        if any(not leg.planted for leg in cat.animator.legs.values()):
            return
    raise AssertionError("no paw lifted")


def test_a_step_that_cannot_finish_is_put_down_short_not_held_forever(monkeypatch):
    """When even standing still is refused (finishing a paw's step would come too close) and so
    is taking the step back, the paw is put down short and the sample is taken (the landing check
    allows LAND_TOL), instead of the cat being held on three legs sample after sample."""
    s, herd, cat = _herd_one(-2.5, 2.4, 0.0)
    _paw_in_the_air(herd, cat, s)
    real = herd.pose_ok
    monkeypatch.setattr(herd, "pose_ok", lambda c, *a, **k: False if c is cat and k.get("pose") is not None
                        else real(c, *a, **k))
    held = cat.held
    herd._advance_sample(cat)
    assert cat.held == held  # taken, not rejected
    assert all(leg.short for leg in cat.animator.legs.values() if not leg.planted)
    s.close()


@pytest.mark.parametrize("phase", [0.15, 0.5, 0.85])
def test_a_withdrawn_step_starts_where_the_paw_is_and_moves_smoothly(phase):
    """put_paws_down(back=True) at an early, middle, or late point of a step: the paw goes back
    to where it lifted from without a jump (its first move is no bigger than PAW_RETURN_SPEED
    allows), no joint turns more than the gait gate's 25 degrees in a 10 ms sample, and it lands
    within a bounded time."""
    from robot_env.cat_motion import PAW_RETURN_SPEED, CatAnimator
    from robot_env.cat_rig import load_rig
    from robot_env.cats import ANIM_PERIOD
    a = CatAnimator(load_rig(), 0)
    a.reset(0.0, 0.0, 0.0)
    x = 0.0
    while True:
        x += 0.3 * ANIM_PERIOD
        a.update(ANIM_PERIOD, x, 0.0, 0.0, 0.3, 0.0)
        lifted = [leg for leg in a.legs.values() if not leg.planted and leg.progress >= phase]
        if lifted:
            break
    leg = lifted[0]
    before, lift_spot = leg.world.copy(), leg.swing_from[:2].copy()
    local0, _ = a.update(0.0, x, 0.0, 0.0, 0.0, 0.0, settle=False, freeze=True)
    local0 = {k: v.copy() for k, v in local0.items()}
    assert a.put_paws_down(back=True)
    previous, prev_local, t = before, local0, 0.0
    while not leg.planted:
        local, _ = a.update(ANIM_PERIOD, x, 0.0, 0.0, 0.0, 0.0, settle=False, freeze=True)
        t += ANIM_PERIOD
        step = float(np.hypot(*(leg.world[:2] - previous[:2])))
        assert step <= 1.6 * PAW_RETURN_SPEED * ANIM_PERIOD + 1e-9  # no jump (smoothstep peaks at 1.5x mean)
        for j, R in local.items():
            if j in prev_local:
                angle = np.degrees(np.arccos(np.clip((np.trace(prev_local[j].T @ R) - 1) / 2, -1, 1)))
                assert angle <= 25.0
        previous, prev_local = leg.world.copy(), {k: v.copy() for k, v in local.items()}
        assert t < 1.0
    assert np.allclose(leg.world[:2], lift_spot)


def _swept_gaps(herd, cat):
    """The gaps of the sample just taken, grown by half of each circle's move over it: what
    _landing_ok and pose_ok check."""
    nxt, prev = cat.next, cat.prev
    inflate = herd._half_chord(cat, herd.circles(cat, nxt.x, nxt.y, nxt.yaw, nxt), herd.circles(cat, pose=prev))
    return herd.gaps(cat, nxt.x, nxt.y, nxt.yaw, nxt, inflate, sweep=True)


@pytest.mark.parametrize("which,outside", [("wall", True), ("wall", False), ("robot", True), ("robot", False),
                                           ("cat", True), ("cat", False)])
def test_a_paw_landing_short_never_goes_below_the_floor(monkeypatch, which, outside):
    """A paw put down short, sample after sample until it is down, with every other move refused
    (each of those samples taken): for the wall, the robot (its comfort gap), and another cat, a
    cat starting just outside the gap never comes (swept over each sample) closer than the gap
    less LAND_TOL, and one starting already below that floor never comes closer at all."""
    from robot_env.cats import LAND_TOL
    k = {"wall": 0, "robot": 1, "cat": 2}[which]
    s = RobotSystem(cats=2, cat_seed=2)
    s.reset(0.8, 2.6, math.pi, (3.0, -3.0))  # the robot parked in the office
    herd = s.cats
    herd.place(1, 30.0, 30.0, 0.0, state="pause")  # the other cat off the floor unless needed
    if which == "wall":
        x, y, yaw, step = 0.43, -3.652, 2.7489, None  # 2 cm from the storage boxes
        limit = WALL_GAP
    elif which == "robot":
        x, y, yaw = 0.8 - 0.62, 2.6, 0.0  # in front of the robot, facing it
        limit = ROBOT_GAP  # the landing's fixed reference, inside it or not
    else:
        herd.place(1, -2.2, 2.4, 0.0, state="pause")
        x, y, yaw = -2.2 - 0.52, 2.4, 0.0  # behind the other cat, facing it
        limit = CAT_GAP
    floor = limit - LAND_TOL
    target = limit + 0.0002 if outside else floor - 0.008
    herd.place(0, x, y, yaw, state="walk")
    cat = herd.cats[0]
    # slide straight along the line that changes the gap until it is at the target
    d = max(((math.cos(a), math.sin(a)) for a in np.linspace(0, 2 * math.pi, 64, endpoint=False)),
            key=lambda d: herd.gaps(cat, x + 0.002 * d[0], y + 0.002 * d[1], yaw)[k])
    sign = 1.0 if herd.gaps(cat, x, y, yaw)[k] < target else -1.0
    for _ in range(20000):
        if (herd.gaps(cat, x, y, yaw)[k] - target) * sign >= 0:
            break
        x, y = x + sign * 0.0002 * d[0], y + sign * 0.0002 * d[1]
    else:
        raise AssertionError("could not place the cat at the wanted gap")
    herd.place(0, x, y, yaw, state="walk")
    cat = herd.cats[0]
    first = herd.gaps(cat, x, y, yaw)[k]
    assert abs(first - target) <= 0.0004 and (first >= floor) == outside  # within a step or two of it
    cat.target_v = 0.3
    leg = cat.animator.legs["RF"]
    leg.planted, leg.progress, leg.swing_from = False, 0.3, leg.world.copy()  # a paw in the air
    real = herd.pose_ok
    monkeypatch.setattr(herd, "pose_ok", lambda c, *a, **kw: False if c is cat and kw.get("pose") is not None
                        else real(c, *a, **kw))
    held, samples = cat.held, 0
    while not leg.planted:  # every sample until the paw is down is a landing, and is taken
        assert samples < 60
        herd._advance_sample(cat)
        herd.time += 0.01
        samples += 1
        assert cat.held == held
        g = _swept_gaps(herd, cat)[k]
        if outside:
            assert g >= floor - 1e-9
        else:
            assert g >= first - 1e-9
    assert leg.planted  # the paw is down
    s.close()


@pytest.mark.slow
def test_a_crowded_cat_finds_its_way_out_on_the_retry(monkeypatch):
    """Regression (roaming seed 4): three cats crowd by the office door; the way-out search of
    the one in the middle finds nothing (t = 224.62 s) and finds one ESCAPE_RETRY later, once
    the cat ahead has moved on, instead of GIVE_WAY_AFTER later; no contact, every gap kept.
    Replayed as recorded, before cats sat down (sitting changes the trajectory): sitting off."""
    from robot_env.cat_motion import CatAnimator
    from robot_env.cats import ESCAPE_RETRY, GIVE_WAY_AFTER
    from robot_env.layout import RoomMap
    monkeypatch.setattr(CatAnimator, "sit_target", property(lambda self: 0.0, lambda self, value: None), raising=False)
    s = RobotSystem(cats=3, cat_seed=4)
    task = RoomMap(s.sim.model).sample_task(1000)
    s.reset(*task.start, task.goal)
    herd = s.cats
    searches = []
    real = herd._escape_search

    def search(c):
        out = real(c)
        if c.index == 1:
            searches.append((herd.time, len(out)))
        return out
    herd._escape_search = search
    s.advance(224.0)
    searches.clear()
    low = [math.inf] * 3
    while herd.time < 227.0:
        s.advance(C.PHYSICS_DT * 20)
        for g in herd.current_gaps():
            low = [min(low[0], g["wall"]), min(low[1], g["robot"]), min(low[2], g["cat"])]
    assert searches[0][1] == 0 and searches[1][1] > 0  # none at first, then a way out
    assert ESCAPE_RETRY - 0.03 <= searches[1][0] - searches[0][0] <= ESCAPE_RETRY + 0.05 < GIVE_WAY_AFTER
    assert s.cat_contacts == 0
    assert low[0] >= WALL_GAP - 0.001 and low[1] >= ROBOT_GAP - 0.001 and low[2] >= CAT_GAP - 0.001
    s.close()


def test_a_cat_in_a_corner_gets_out_using_its_cushion():
    """Regression (roaming seed 14): in the office corner, facing into it, every move comes a
    little closer to a wall, so inside the planner's cushion nothing was feasible and the cat stood
    for 79 s. A cat stuck like this may use the cushion (down to the room a root move keeps), so
    it gets out, with every gap still kept."""
    import math as _m
    from robot_env.cats import CAT_GAP, ROBOT_GAP, WALL_GAP
    s, herd, cat = _herd_one(-4.175, 3.938, _m.radians(26.4))
    s.reset(4.0, -4.0, 0.0, (3.0, -3.0))  # the robot far away
    herd.place(0, -4.175, 3.938, _m.radians(26.4), state="walk")
    cat = herd.cats[0]
    cat.state_until, cat.target_v, cat.target_yaw = herd.time + 30.0, 0.15, _m.radians(225.0)
    start = (cat.x, cat.y)
    low = [_m.inf] * 3
    for _ in range(int(5.0 / 0.02)):
        s.advance(0.02)
        g = herd.current_gaps()[0]
        low = [min(low[0], g["wall"]), min(low[1], g["robot"]), min(low[2], g["cat"])]
    assert _m.hypot(cat.x - start[0], cat.y - start[1]) > 0.05
    assert low[0] >= WALL_GAP - 0.001 and low[1] >= ROBOT_GAP - 0.001 and low[2] >= CAT_GAP - 0.001
    s.close()
