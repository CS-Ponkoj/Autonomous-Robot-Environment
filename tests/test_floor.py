"""The indoor office floor: geometry, doorways, map, sensing, and a safety stress test."""

import math

import mujoco
import numpy as np
import pytest

from robot_env import config as C
from robot_env.layout import RoomMap, region_of
from robot_env.sim import RobotSim
from robot_env.system import RobotSystem
from robot_env.types import Command

DOORS = [  # (name, center x, center y, axis of the wall)
    ("office", -2.5, 0.75, "x"),
    ("lab", 2.5, 0.75, "x"),
    ("storage", -3.0, -0.75, "x"),
    ("reception", 2.0, -0.75, "x"),
    ("office_lab", 0.0, 3.5, "y"),
]


@pytest.fixture(scope="module")
def sim():
    s = RobotSim()
    yield s
    s.close()


@pytest.fixture(scope="module")
def room(sim):
    return RoomMap(sim.model)


def test_visible_and_colliding_geometry_agree(sim):
    """World geoms: everything that collides is visible as a solid (group 0), and every
    solid-group geom collides. Visual detail (group 2) and the ceiling (3) never collide."""
    m = sim.model
    for g in range(m.ngeom):
        if m.body_rootid[m.geom_bodyid[g]] == sim.robot_body:
            continue
        name = m.geom(g).name
        if m.geom_contype[g] != 0:
            assert m.geom_group[g] == 0, name
        if m.geom_group[g] == 0:
            assert m.geom_contype[g] != 0, name
        if m.geom_group[g] in (2, 3):
            assert m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0, name


@pytest.mark.parametrize("name,x,y,axis", DOORS)
def test_doorways_are_clear_and_fit_the_robot(sim, room, name, x, y, axis):
    # At lidar height, a ray straight through the doorway's center meets nothing for 1 m.
    origin = np.array([x, y, 0.135]) - (np.array([0, 0.5, 0]) if axis == "x" else np.array([0.5, 0, 0]))
    direction = np.array([0, 1.0, 0]) if axis == "x" else np.array([1.0, 0, 0])
    hit = sim.ray_to_solid(origin, direction)
    assert hit < 0 or hit > 1.0, (name, hit)
    # The inflated planning map has free cells through the doorway.
    assert room.free[room.cell(x, y)], name
    # Clear width between the jambs is 0.9 m and fits the inflated robot.
    assert 0.9 >= 2 * (C.CIRCUMSCRIBED_RADIUS + C.PLANNING_CLEARANCE) + C.GRID_RESOLUTION


def test_config_rooms_match_the_generated_walls(sim):
    m = sim.model
    north = [g for g in range(m.ngeom) if m.geom(g).name.startswith("corridor_north_") and "door" not in m.geom(g).name]
    south = [g for g in range(m.ngeom) if m.geom(g).name.startswith("corridor_south_") and "door" not in m.geom(g).name]
    north_face = m.geom_pos[north[0]][1] - m.geom_size[north[0]][1]
    south_face = m.geom_pos[south[0]][1] + m.geom_size[south[0]][1]
    assert C.ROOMS["corridor"][3] == pytest.approx(north_face)
    assert C.ROOMS["corridor"][2] == pytest.approx(south_face)
    assert C.ROOMS["office"][2] == pytest.approx(north_face + 2 * m.geom_size[north[0]][1])
    assert C.ROOMS["storage"][3] == pytest.approx(south_face - 2 * m.geom_size[south[0]][1])
    east = m.geom("wall_east_0")
    assert east.pos[0] - east.size[0] == pytest.approx(C.FLOOR_HALF_SIZE)


def test_every_region_is_reachable_from_the_corridor(room):
    for name, (x0, x1, y0, y1) in C.ROOMS.items():
        xs, ys = np.meshgrid(np.linspace(x0 + 0.3, x1 - 0.3, 15), np.linspace(y0 + 0.3, y1 - 0.3, 15))
        free = [(x, y) for x, y in zip(xs.ravel(), ys.ravel()) if room.free[room.cell(x, y)]]
        assert free, name
        assert any(room.find_path((0.0, 0.0), p) for p in free[:10]), name


def test_tasks_always_cross_regions(room):
    for seed in C.HELDOUT_SEEDS:
        t = room.sample_task(seed)
        assert region_of(*t.start[:2]) != region_of(*t.goal)


def test_map_includes_solids_on_static_child_bodies():
    """Regression: geoms on a static body other than the world were ignored."""
    extra = ('<body name="shelf_body" pos="1.0 1.5 0"><geom name="shelf_on_body" type="box" '
             'pos="0 0 0.3" size="0.3 0.3 0.3"/></body>')
    s = RobotSim(include_obstacles=False, extra_world_xml=extra)
    room = RoomMap(s.model)
    assert room.point_clearance(1.0, 1.5) == 0.0
    s.close()


def test_map_ignores_overhead_parts_like_door_headers(room):
    # Door headers span the doorway above 2.1 m; the doorway center must stay free.
    assert room.point_clearance(-2.5, 0.75) > 0.3


@pytest.mark.parametrize("pose,deg,expected", [
    ((1.0, 2.5, 0.0), 0, 1.0),          # lab island bench west face at x = 2.0
    ((-2.5, 2.0, -math.pi / 2), 0, None),  # looking out through the office doorway: no wall right ahead
    ((-0.45, 1.6, math.pi / 2), 0, 4.62 - 0.3 - 1.6),  # office cabinet front face (y = 4.32)
])
def test_lidar_sees_floor_geometry(pose, deg, expected):
    s = RobotSystem()
    s.reset(pose[0], pose[1], pose[2], (0.0, 0.0))
    s.advance(0.04)
    obs = s.observe()
    i = int(np.argmin(np.abs(np.degrees(obs.lidar_angles) - deg)))
    if expected is None:
        assert obs.lidar[i] > 1.4  # through the door and across the corridor
    else:
        assert obs.lidar_valid[i] and obs.lidar[i] == pytest.approx(expected, abs=0.05)
    s.close()


def test_lidar_ignores_the_ceiling_and_visual_detail(sim):
    origin = np.array([0.0, 0.0, 0.135])
    geomid = np.zeros(1, dtype=np.int32)
    lidar_groups = np.array([1, 1, 0, 0, 1, 1], dtype=np.uint8)
    hit = mujoco.mj_ray(sim.model, sim.data, origin, np.array([0, 0, 1.0]), lidar_groups, 1, sim.robot_body, geomid)
    assert hit < 0  # straight up: the ceiling (group 3) is not seen
    hit = mujoco.mj_ray(sim.model, sim.data, origin, np.array([0, 0, 1.0]), np.ones(6, dtype=np.uint8), 1,
                        sim.robot_body, geomid)
    assert hit == pytest.approx(C.WALL_HEIGHT - 0.135, abs=0.01)  # it is there for cameras


def test_safety_stress_on_the_office_floor():
    """Random poses near walls, door jambs, and furniture, with random mixed commands
    (turning while driving, reversing while turning, sudden stops): never a collision."""
    s = RobotSystem()
    room = RoomMap(s.sim.model)
    rng = np.random.default_rng(2026)
    commands = [(0.5, 0.0), (0.3, 0.0), (-0.15, 0.0), (-0.25, 0.0), (0.0, 1.5), (0.0, -1.5),
                (0.5, 1.5), (0.5, -1.5), (-0.25, 1.0), (-0.25, -1.0), (0.3, 0.6), (0.0, 0.0)]
    for _ in range(150):
        while True:
            x, y = rng.uniform(-4.85, 4.85, 2)
            if C.CIRCUMSCRIBED_RADIUS + 0.06 < room.point_clearance(x, y) < 0.6:
                break
        s.reset(x, y, rng.uniform(-math.pi, math.pi), (0.0, 0.0))
        for _ in range(int(rng.integers(4, 12))):
            v, w = commands[rng.integers(len(commands))]
            for _ in range(int(rng.integers(1, 6))):
                s.drive(v, w)
                s.advance(0.1)
        assert s.collisions == 0, (x, y)
    s.close()


def test_robot_colliders_and_mass_are_unchanged_by_the_new_look():
    """The visual redesign must not change what collides or the calibrated mass."""
    s = RobotSim()
    m = s.model
    assert m.geom("chassis").size.tolist() == pytest.approx([0.15, 0.10, 0.035])
    # Tire collision is a narrow single-point ellipsoid with a stiff contact that wins over the
    # floor's (agreed physics fixes); wheel mass and inertia stay pinned to the 3 cm wheel.
    import mujoco
    for tire in ("left_tire", "right_tire"):
        g = m.geom(tire)
        assert g.type[0] == mujoco.mjtGeom.mjGEOM_ELLIPSOID
        assert g.size.tolist() == pytest.approx([0.05, 0.05, 0.008])
        assert g.priority[0] == 1 and g.solref.tolist() == pytest.approx([0.002, 1.0])
        assert g.friction.tolist() == pytest.approx([1.2, 0.005, 0.0001])
    for caster in ("front_caster", "rear_caster"):  # soft, damped (sprung) caster contact
        assert m.geom(caster).solref.tolist() == pytest.approx([0.1, 3.0]) and m.geom(caster).priority[0] == 1
    assert m.actuator_forcerange.ravel().tolist() == pytest.approx([-0.3, 0.3, -0.3, 0.3])  # small gear motors
    wheel = m.body("left_wheel").id
    assert m.body_mass[wheel] == pytest.approx(0.2)
    assert m.body_inertia[wheel].tolist() == pytest.approx([0.00014, 0.00025, 0.00014])
    assert m.opt.cone == 1 and m.opt.impratio == pytest.approx(1.0)  # elliptic friction cone
    assert m.body_subtreemass[s.robot_body] == pytest.approx(2.0 + 0.05 + 0.05 + 0.1 + 0.2 + 0.2)
    colliding = [m.geom(g).name for g in range(m.ngeom)
                 if m.body_rootid[m.geom_bodyid[g]] == s.robot_body and m.geom_contype[g] != 0]
    assert sorted(colliding) == sorted(["chassis", "front_caster", "rear_caster", "lidar_housing", "left_tire", "right_tire"])
    s.close()
    assert Command(0, 0) == Command()


def test_world_xml_matches_its_generator():
    """world.xml is generated: it must equal what tools/build_world.py produces now."""
    import importlib.util
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location("build_world", root / "tools" / "build_world.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert (root / "robot_env" / "world.xml").read_text(encoding="utf-8") == module.generate()


def test_every_robot_part_stays_inside_the_physical_envelope():
    """V2: visual detail must not make the robot look bigger than it is. Every robot geom's
    bounding box stays inside the safety footprint (x, y), above the floor, below 0.16 m."""
    s = RobotSim()
    s.reset(0.0, 0.0, 0.0, (2.0, 2.0))
    m, d = s.model, s.data
    body_z = d.xpos[s.robot_body][2]
    for g in range(m.ngeom):
        if m.body_rootid[m.geom_bodyid[g]] != s.robot_body:
            continue
        center, half = m.geom_aabb[g][:3], m.geom_aabb[g][3:]
        rot = d.geom_xmat[g].reshape(3, 3)
        world_center = d.geom_xpos[g] + rot @ center
        extent = np.abs(rot) @ half  # world-axis half extents of the rotated box
        lo, hi = world_center - extent, world_center + extent
        name = m.geom(g).name or f"geom {g}"
        assert hi[0] <= C.FOOTPRINT_HALF_LENGTH + 1e-4 and lo[0] >= -C.FOOTPRINT_HALF_LENGTH - 1e-4, name
        assert hi[1] <= C.FOOTPRINT_HALF_WIDTH + 1e-4 and lo[1] >= -C.FOOTPRINT_HALF_WIDTH - 1e-4, name
        assert lo[2] >= body_z - 0.0515 and hi[2] <= 0.16, name
    s.close()


def test_tire_contacts_are_single_points_and_the_solver_is_quiet():
    """Pivoting in the open and next to a wall: exactly one floor contact per tire at every
    physics step. A mixed drive (starts, arcs, pivots) raises no solver warnings."""
    from robot_env.system import RobotSystem
    s = RobotSystem()
    m = s.sim.model
    tires = (m.geom("left_tire").id, m.geom("right_tire").id)
    floor = m.geom("floor").id

    def contacts(d):
        """Per tire: (floor contacts, contacts with anything else)."""
        per = {t: [0, 0] for t in tires}
        for c in range(d.ncon):
            g1, g2 = d.contact[c].geom1, d.contact[c].geom2
            for tire, other in ((g1, g2), (g2, g1)):
                if tire in per:
                    per[tire][0 if other == floor else 1] += 1
        return per

    for pose in [(-4.0, 0.0, 0.0), (4.6, 0.0, 3.1416), (4.6, 0.0, 0.0)]:  # corridor; end wall behind / ahead
        s.reset(*pose, (0.0, 0.0))
        landed = None
        for i in range(int(1.5 / C.PHYSICS_DT)):
            if i % 40 == 0:
                s.drive(0.0, 1.0 if (i // 1000) % 2 == 0 else -1.5)
            s.advance(C.PHYSICS_DT)
            per = contacts(s.sim.data)
            if landed is None:
                if all(f == 1 for f, _ in per.values()):
                    landed = i  # reset places the robot just above the floor
                continue
            assert all(f == 1 and o == 0 for f, o in per.values()), (pose, i, per)
        assert landed is not None and landed * C.PHYSICS_DT <= 0.02, landed  # lands in about 10 ms
        s.reset(*pose, (0.0, 0.0))
        for i in range(int(2.0 / C.PHYSICS_DT)):
            if i % 40 == 0:
                s.drive(0.3 if (i // 400) % 2 else 0.0, 1.0)
            s.advance(C.PHYSICS_DT)
        assert sum(int(w.number) for w in s.sim.data.warning) == 0
    s.close()


def test_config_doors_match_the_generated_door_frames(sim):
    """config.DOORS (where cats never rest) is the centre of every door frame in the world."""
    m = sim.model
    names = [m.geom(g).name for g in range(m.ngeom)]
    centres = []
    for name in names:
        if name.endswith("_jambL"):
            a, b = m.geom(name).pos, m.geom(name[:-1] + "R").pos
            centres.append(((a[0] + b[0]) / 2, (a[1] + b[1]) / 2))
    assert len(centres) == len(C.DOORS)
    for door in C.DOORS:
        assert min(math.hypot(door[0] - x, door[1] - y) for x, y in centres) < 1e-6, door
