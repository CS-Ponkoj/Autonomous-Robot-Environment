"""The furniture: detailed meshes agree with their solid members, the lidar sees everything the
robot can touch, and the robot driven straight at any piece of furniture stops without touching it."""

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest

from robot_env import config as C
from robot_env.layout import RoomMap
from robot_env.system import RobotSystem

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))


def _tool(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def check():
    fc = _tool("furniture_check")
    sim = fc.compiled_world()
    yield fc, sim
    sim.close()


def test_the_lidar_sees_everything_the_robot_can_touch(check):
    """Every solid static geom below the robot's top spans the lidar plane (1 cm margins) or lies
    inside solids that do (2 cm): the single-plane lidar always meets a seen solid first."""
    fc, sim = check
    top, plane = fc.robot_top_and_plane(sim.model, sim.data)
    # A tripwire, not a tolerance: the lidar rule was agreed for this robot (lidar housing top and
    # scan plane both at 0.1355 m). A different robot must revisit the furniture.
    assert top == pytest.approx(0.1355, abs=1e-4) and plane == pytest.approx(0.1355, abs=1e-4)
    assert fc.lidar_gate(sim.model, sim.data, top, plane) == []


def test_every_furniture_mesh_agrees_with_its_members(check):
    fc, sim = check
    top, _ = fc.robot_top_and_plane(sim.model, sim.data)
    items = fc.build_world.furniture_items
    assert items
    for it in items:
        r = fc.check_item(sim.model, sim.data, it, max(0.40, top + 0.05), top, sight=False)
        assert r["pass"], r


def test_nothing_drawn_passes_into_anything_else(check):
    """No furniture mesh passes into another mesh (its own item's too), a wall, a door, or another
    item's solids; and the check does catch it (a lamp moved 4 cm so its shade enters the wall, a
    planter sunk 3 cm into its stand)."""
    import copy
    fc, sim = check
    items = fc.build_world.furniture_items
    loose = [m for m in fc.build_world.placed_models if m["item"] is None]  # the wall and ceiling items
    assert loose and fc.overlap_gate(sim.model, sim.data, items, extra_models=loose) == []
    for name, model_id, axis, delta in (("office_lamp", "proc_floor_lamp", 0, -0.04),
                                        ("office_plant", "potted_plant_01", 2, -0.03)):
        moved = copy.deepcopy(items)
        for it in moved:
            for m in it["meshes"]:
                if it["name"] == name and m["model"] == model_id:
                    m["pos"][axis] += delta
        assert any(f["item"] == name for f in fc.overlap_gate(sim.model, sim.data, moved)), name


@pytest.fixture(scope="module")
def scene():
    """Each furniture item's centre (its members' mean), and the room map."""
    s = RobotSystem()
    room = RoomMap(s.sim.model)
    bw = _tool("build_world")
    bw.generate()
    m, d = s.sim.model, s.sim.data
    centres = {}
    for it in bw.furniture_items:
        pts = np.array([d.geom_xpos[m.geom(n).id][:2] for n in it["members"]])
        centres[it["name"]] = pts.mean(0)
    s.close()
    return centres, room


def _starts(room, centre, bearings=16):
    """Free start points about 1 to 2 m from an item, from each of `bearings` directions where the
    inflated robot fits, facing it."""
    out = []
    for k in range(bearings):
        a = 2 * math.pi * k / bearings
        for r in np.arange(1.0, 2.05, 0.1):
            x, y = centre[0] + r * math.cos(a), centre[1] + r * math.sin(a)
            if abs(x) < 4.8 and abs(y) < 4.8 and room.free[room.cell(x, y)]:
                out.append((x, y, math.atan2(centre[1] - y, centre[0] - x), r))
                break
    return out


@pytest.mark.parametrize("level", range(len(C.SPEED_LEVELS)), ids=[f"level{i + 1}" for i in range(len(C.SPEED_LEVELS))])
def test_driving_straight_at_furniture_never_touches_it(scene, level):
    """At every speed level, from every free one of 16 directions, the robot driven straight at each
    furniture item stops (the safety layer) without contact."""
    v, _ = C.SPEED_LEVELS[level]
    centres, room = scene
    s = RobotSystem()
    failed = []
    for name, centre in centres.items():
        starts = _starts(room, centre)
        assert len(starts) >= 2, name  # (an item in a corner can only be approached from a few sides)
        for x, y, yaw, r in starts:
            s.reset(x, y, yaw, (0.0, 0.0))
            for _ in range(int((r + 0.5) / v / 0.1) + 15):
                s.drive(v, 0.0)
                s.advance(0.1)
            if s.collisions:
                failed.append((name, round(x, 2), round(y, 2)))
    s.close()
    assert not failed, failed


@pytest.mark.parametrize("start,target", [
    ((-2.915, 3.533), (-3.778, 3.39)),   # chassis met a star-base chair leg (z 0.067 m)
    ((-2.635, 3.617), (-3.494, 3.629)),  # chassis met a caster (z 0.05 m)
    ((-4.248, 3.268), (-4.716, 3.557)),  # front caster met a 3 cm lamp base
])
def test_regression_the_level4_furnished_stress_contacts(start, target):
    """The three level-4 contacts found when the office got real furniture (parts below the lidar
    plane): driving from those start poses at the contact points, at level 4, touches nothing."""
    v, w = C.SPEED_LEVELS[3]
    s = RobotSystem()
    yaw = math.atan2(target[1] - start[1], target[0] - start[0])
    for turn in (0.0, w / 2, -w / 2):
        s.reset(start[0], start[1], yaw, (0.0, 0.0))
        for _ in range(40):
            s.drive(v, turn)
            s.advance(0.1)
        assert s.collisions == 0, (start, turn)
    s.close()


MOVING = ("walk", "travel", "dart", "flee", "yield")  # as tools/roam_check.py


def _longest_stall(pose, seconds=15.0, robot=(4.0, -4.0, 0.0), goal=(3.0, -3.0)):
    """A cat placed walking at pose: the longest time it wanted to move (MOVING) without moving
    5 cm, sampled every 0.5 s like tools/roam_check.py; and its lowest wall, robot, cat gaps."""
    s = RobotSystem(cats=1, cat_seed=2)
    s.reset(*robot, goal)
    herd = s.cats
    herd.place(0, pose[0], pose[1], math.radians(pose[2]), state="walk")
    cat = herd.cats[0]
    first = herd.current_gaps()[0]["wall"]
    ref, stall, low = None, 0.0, [math.inf] * 3
    for _ in range(int(seconds / 0.5)):
        s.advance(0.5)
        g = herd.current_gaps()[0]
        low = [min(low[0], g["wall"]), min(low[1], g["robot"]), min(low[2], g["cat"])]
        if cat.state not in MOVING or cat.frozen:
            ref = None
        elif ref is None or math.hypot(cat.x - ref[1], cat.y - ref[2]) > 0.05:
            ref = (herd.time, cat.x, cat.y)
        else:
            stall = max(stall, herd.time - ref[0])
    s.close()
    return first, stall, low


@pytest.mark.parametrize("pose", [(-4.13, 4.07, 60.0), (-3.41, 3.95, 300.0), (-3.8, 4.4, 90.0),
                                  (-4.65, 3.2, 225.0), (-3.3, 3.7, 180.0), (-3.0, 3.8, 45.0)])
def test_a_cat_never_gets_stuck_by_the_office_furniture(pose):
    """Spot checks (the full sweep of 615 tight poses around the desk is tools/cat_pocket_sweep.py):
    with the chair pulled out, cats placed in the pockets between the chair, the pedestals, and the
    desk front stood still for 60 s. With the chair tucked in at the kneehole and the lamp in the
    corner, a cat placed walking at these poses (the old pockets, under the desk, the lamp corner,
    beside the chair) never wants to move without moving for more than 5 s, the roam's stall rule,
    and keeps every gap. (One sweep pose is under the chair's seat, boxed in by its legs, which are
    0.385 m apart: less than twice the cats' 0.2 m navigation clearance, so no cat walks in there.)"""
    from robot_env.cats import CAT_GAP, ROBOT_GAP, WALL_GAP
    first, stall, low = _longest_stall(pose)
    assert first >= WALL_GAP  # a pose a cat can really be in
    assert stall <= 5.0 + 1e-6, stall  # (the 0.5 s samples add up with float error: 5 s is 5 s)
    assert low[0] >= WALL_GAP - 0.001 and low[1] >= ROBOT_GAP - 0.001 and low[2] >= CAT_GAP - 0.001


@pytest.mark.parametrize("pose", [(-3.45, 3.91, 90.0), (-4.45, 3.83, 150.0)])
def test_the_old_pockets_beside_the_chair_and_lamp_are_closed_to_cats(pose):
    """Two former pockets (between the chair and the right pedestal; between the desk's west end and
    the lamp) are now too small for a cat: none can stand there with its wall gap."""
    from robot_env.cats import WALL_GAP
    s = RobotSystem(cats=1, cat_seed=2)
    s.reset(4.0, -4.0, 0.0, (3.0, -3.0))
    s.cats.place(0, pose[0], pose[1], math.radians(pose[2]), state="pause")
    assert s.cats.current_gaps()[0]["wall"] < WALL_GAP
    s.close()


OFFICE_OPEN = (-4.0, 1.4, 0.0)  # the robot parked in the office, far from the reception


@pytest.mark.parametrize("pose", [(1.124, -3.90, 0.0),     # between the sofa and the coffee table (0.45 m)
                                  (1.124, -3.415, 90.0),  # under the coffee table, between its legs
                                  (4.39, -4.25, 270.0),   # into the gap between the ottoman and the lamp (0.52 m)
                                  (3.00, -3.95, 0.0),     # in front of the side table and the armchair
                                  (0.32, -3.40, 270.0)])  # between the coffee table and the west wall
def test_a_cat_never_gets_stuck_by_the_reception_furniture(pose):
    """The reception's tight places (gaps of 0.4 to 0.6 m, and the space under the coffee table): a
    cat placed walking there never wants to move without moving for more than 5 s, the roam's
    stall rule, and keeps every gap (the robot parked in the office)."""
    from robot_env.cats import CAT_GAP, ROBOT_GAP, WALL_GAP
    first, stall, low = _longest_stall(pose, robot=OFFICE_OPEN, goal=(-4.0, 2.4))
    assert first >= WALL_GAP  # a pose a cat can really be in
    assert stall <= 5.0, stall
    assert low[0] >= WALL_GAP - 0.001 and low[1] >= ROBOT_GAP - 0.001 and low[2] >= CAT_GAP - 0.001


@pytest.mark.parametrize("pose", [(2.669, -4.75, 90.0),   # between the side table and the armchair (3 cm)
                                  (3.519, -4.70, 90.0),   # between the armchair and the ottoman (3 cm)
                                  (4.93, -4.93, 45.0),    # behind the lamp in the corner
                                  (4.93, -3.0, 90.0),     # between the bin and the east wall
                                  (4.91, -1.2, 90.0)])    # between the plant stand and the east wall
def test_the_reception_gaps_below_a_cat_are_closed(pose):
    """The reception's gaps below 0.23 m are too small for a cat: none can stand there with its
    wall gap."""
    from robot_env.cats import WALL_GAP
    s = RobotSystem(cats=1, cat_seed=2)
    s.reset(*OFFICE_OPEN, (-4.0, 2.4))
    s.cats.place(0, pose[0], pose[1], math.radians(pose[2]), state="pause")
    assert s.cats.current_gaps()[0]["wall"] < WALL_GAP
    s.close()


def test_every_furniture_file_matches_its_manifest():
    """robot_env/assets/furniture/MANIFEST.json lists every file there with its SHA-256 and size,
    and nothing else is there: the committed meshes, textures, and records are exactly the ones
    tools/fetch_models.py and tools/proc_furniture.py made (line endings included)."""
    import hashlib
    import json
    folder = ROOT / "robot_env" / "assets" / "furniture"
    manifest = json.loads((folder / "MANIFEST.json").read_text(encoding="utf-8"))
    on_disk = {p.name for p in folder.iterdir() if p.name != "MANIFEST.json"}
    assert on_disk == set(manifest["files"])
    for name, entry in manifest["files"].items():
        data = (folder / name).read_bytes()
        assert len(data) == entry["bytes"] and hashlib.sha256(data).hexdigest() == entry["sha256"], name
    for rec in manifest["models"]:
        assert rec["license"] == "CC0 1.0" and rec["source"].startswith("https://polyhaven.com/a/"), rec["id"]
