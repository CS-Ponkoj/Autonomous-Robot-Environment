"""The numeric kernels (robot_env/kernels.py): each one gives the same results compiled and
interpreted (the same source run as plain Python), and the same as the plain NumPy / Python
implementation it replaced, on random inputs and on inputs at the edges (raster borders, points
exactly at a limit, empty sets, degenerate capsules, unreachable leg targets)."""

from __future__ import annotations

import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from robot_env import config as C
from robot_env import kernels as K

ROOT = Path(__file__).resolve().parent.parent
RNG = np.random.default_rng(1234)


def both(kernel, *args):
    """Run a kernel compiled and interpreted on copies of the same arguments; returns both
    (results, arguments after the call), so in-place outputs can be compared too."""
    def copy(a):
        return a.copy() if isinstance(a, np.ndarray) else a
    a1 = [copy(a) for a in args]
    a2 = [copy(a) for a in args]
    r1 = kernel(*a1)
    r2 = getattr(kernel, "py_func", kernel)(*a2)
    return (r1, a1), (r2, a2)


def same(x, y):
    """Exactly equal (arrays, tuples, scalars)."""
    if isinstance(x, tuple):
        return all(same(a, b) for a, b in zip(x, y))
    if isinstance(x, np.ndarray):
        return np.array_equal(x, y, equal_nan=True)
    return x == y or (x != x and y != y)


def assert_same(kernel, *args):
    (r1, a1), (r2, a2) = both(kernel, *args)
    assert same(r1, r2)
    for x, y in zip(a1, a2):
        if isinstance(x, np.ndarray):
            assert np.array_equal(x, y, equal_nan=True)
    return r1, a1


# ----- skeleton -----

def _animator():
    from robot_env.cat_motion import CatAnimator
    from robot_env.cat_rig import load_rig
    a = CatAnimator(load_rig(), 0)
    a.reset(0.0, 0.0, 0.0)
    a._fk({})
    return a


def _random_local(a, k=12, spread=0.4):
    from robot_env.cat_motion import rot_x, rot_y, rot_z
    n = len(a.rig.joint_names)
    return {int(j): rot_z(RNG.normal(0, spread)) @ rot_y(RNG.normal(0, spread)) @ rot_x(RNG.normal(0, spread))
            for j in RNG.choice(n, k, replace=False)}


def _fk_reference(a, local):
    """Forward kinematics with NumPy, one hierarchy level at a time (the replaced code)."""
    rig = a.rig
    L = a._local_array(local)
    pos, rot = rig.bind_pos.copy(), L.copy()
    for lv, ks in enumerate(a._levels):
        if lv == 0:
            continue
        par = rig.parents[ks]
        pos[ks] = pos[par] + (rot[par] @ a._offsets[ks][:, :, None])[:, :, 0]
        rot[ks] = rot[par] @ L[ks]
    return pos, rot


def test_fk_kernel_compiled_interpreted_and_reference():
    a = _animator()
    for _ in range(200):
        local = _random_local(a)
        L = a._local_array(local)
        _, (o, p, offs, LL, pos, rot) = assert_same(K.fk, a._fk_order, a._parents, a._offsets, L,
                                                    a.rig.bind_pos.copy(), L.copy())
        ref_pos, ref_rot = _fk_reference(a, local)
        assert np.allclose(pos, ref_pos, rtol=0, atol=1e-12) and np.allclose(rot, ref_rot, rtol=0, atol=1e-12)


def test_the_legs_only_pass_equals_the_full_pass():
    """After the IK, only the legs' joints (and below) are recomputed: exactly the full result."""
    a = _animator()
    for _ in range(200):
        local = _random_local(a)
        legs = {k: v for k, v in local.items() if k in set(a._leg_order.tolist())}
        rest = {k: v for k, v in local.items() if k not in legs}
        pos, rot = a._fk(rest)
        part = a._fk(local, pos, rot)
        full = a._fk(local)
        assert np.array_equal(part[0], full[0]) and np.array_equal(part[1], full[1])


def test_leg_ik_kernel_matches_the_reference_and_itself():
    a = _animator()
    for trial in range(300):
        pos, rot = a._fk(_random_local(a, spread=0.15))
        for leg in a.legs.values():
            # reachable targets, and some far out of reach (stretched) or tucked in
            scale = (0.01, 0.03, 0.2)[trial % 3]
            target = pos[leg.paw] + RNG.normal(0, scale, 3)
            ref = a._leg_ik_reference(leg, target, pos, rot)
            got = a._leg_ik(leg, target, pos, rot)
            assert got.keys() == ref.keys()
            for k in got:
                assert np.allclose(got[k], ref[k], rtol=0, atol=1e-9)
            assert_same(K.leg_ik, target, pos, rot, a.rig.bind_pos, a._parents,
                        -1 if leg.scapula is None else leg.scapula, leg.upper, leg.lower, leg.paw,
                        np.empty((4, 3, 3)))


def test_ieee_remainder_matches_math_remainder():
    for x in list(RNG.uniform(-20, 20, 2000)) + [math.pi, -math.pi, 3 * math.pi, 0.0, 2 * math.pi]:
        assert K.ieee_remainder(float(x), 2 * math.pi) == pytest.approx(math.remainder(x, 2 * math.pi), abs=1e-12)


# ----- clearance and gaps -----

def _grid():
    from robot_env.cats import RASTER, RASTER_SLACK, _clearance_raster
    from robot_env.system import RobotSystem
    s = RobotSystem(cats=0)
    grid, origin = _clearance_raster(s.sim.model)
    s.close()
    return grid, float(origin), RASTER, RASTER_SLACK


def _clearance_reference(grid, origin, cell, slack, px, py):
    """The replaced NumPy bilinear clearance."""
    u, v = (px - origin) / cell, (py - origin) / cell
    i, j = np.floor(u).astype(int), np.floor(v).astype(int)
    ok = (i >= 0) & (i < grid.shape[0] - 1) & (j >= 0) & (j < grid.shape[1] - 1)
    out = np.full(np.shape(i), -1.0)
    i, j, fu, fv = i[ok], j[ok], (u - np.floor(u))[ok], (v - np.floor(v))[ok]
    out[ok] = ((1 - fu) * (1 - fv) * grid[i, j] + fu * (1 - fv) * grid[i + 1, j]
               + (1 - fu) * fv * grid[i, j + 1] + fu * fv * grid[i + 1, j + 1]) - slack
    return out


def test_clearance_kernel_at_nodes_edges_and_off_the_floor():
    grid, origin, cell, slack = _grid()
    far = origin + cell * (grid.shape[0] - 1)
    nodes = origin + cell * np.arange(0, grid.shape[0], 37)
    px = np.concatenate([RNG.uniform(origin - 0.5, far + 0.5, 3000), nodes, [origin, far, far - 1e-12, origin - 1e-12]])
    py = np.concatenate([RNG.uniform(origin - 0.5, far + 0.5, 3000), nodes[::-1], [origin, far, origin, far]])
    _, args = assert_same(K.clearance, grid, origin, cell, slack, px, py, np.empty(len(px)))
    ref = _clearance_reference(grid, origin, cell, slack, px, py)
    assert np.allclose(args[-1], ref, rtol=0, atol=1e-12)
    for x, y in zip(px[:200], py[:200]):
        assert_same(K.clearance_at, grid, origin, cell, slack, float(x), float(y))


def test_gaps_kernel_with_no_other_cats_zero_and_large_inflation():
    grid, origin, cell, slack = _grid()
    for others in (0, 1, 3):
        for grow in (0.0, 0.05, 1.0):
            circ = np.column_stack([RNG.uniform(-4, 4, 20), RNG.uniform(-4, 4, 20), RNG.uniform(0.02, 0.08, 20)])
            oc = np.array([np.column_stack([RNG.uniform(-4, 4, 20), RNG.uniform(-4, 4, 20), RNG.uniform(0.02, 0.08, 20)])
                           for _ in range(others)]).reshape(others, 20, 3)
            og = RNG.uniform(0, 0.01, (others, 20))
            _, args = assert_same(K.gaps, circ, np.full(20, grow), grid, origin, cell, slack, 0.5, -0.5,
                                  C.CIRCUMSCRIBED_RADIUS, oc, og, np.ones(others, dtype=np.bool_), np.empty(3))
            wall, robot, other = args[-1]
            r = circ[:, 2] + grow
            assert wall == pytest.approx(float((_clearance_reference(grid, origin, cell, slack, circ[:, 0], circ[:, 1]) - r).min()), abs=1e-12)
            assert robot == pytest.approx(float((np.hypot(circ[:, 0] - 0.5, circ[:, 1] + 0.5) - r - C.CIRCUMSCRIBED_RADIUS).min()), abs=1e-12)
            if others:
                d = np.hypot(circ[None, :, None, 0] - oc[:, None, :, 0], circ[None, :, None, 1] - oc[:, None, :, 1])
                assert other == pytest.approx(float((d - r[None, :, None] - (oc[:, None, :, 2] + og[:, None, :])).min()), abs=1e-12)
            else:
                assert other == math.inf


def test_plan_gaps_kernel_with_and_without_other_cats():
    grid, origin, cell, slack = _grid()
    for others in (0, 2):
        X, Y, Th = RNG.uniform(-3, 3, (47, 12)), RNG.uniform(-3, 3, (47, 12)), RNG.uniform(-4, 4, (47, 12))
        lx, ly, r = RNG.normal(0, 0.15, 20), RNG.normal(0, 0.05, 20), RNG.uniform(0.02, 0.08, 20)
        times = 0.05 * np.arange(1, 13)
        q = 20 * others
        ox, oy, orad, ovx, ovy = (RNG.uniform(-3, 3, q), RNG.uniform(-3, 3, q), RNG.uniform(0.02, 0.08, q),
                                  RNG.normal(0, 0.3, q), RNG.normal(0, 0.3, q))
        prx, pry = 1.0 + 0.3 * times, -1.0 + 0.0 * times
        assert_same(K.plan_gaps, X, Y, Th, lx, ly, r, grid, origin, cell, slack, 0.03, prx, pry, 0.48,
                    ox, oy, orad, ovx, ovy, times, 0.05, np.empty((47, 12)))


def test_place_circles_and_half_chord():
    local = RNG.normal(0, 0.2, (20, 3))
    radius = RNG.uniform(0.02, 0.08, 20)
    _, args = assert_same(K.place_circles, local, radius, 1.5, -2.0, 2.2, np.empty((20, 3)))
    c, s = math.cos(2.2), math.sin(2.2)
    assert np.allclose(args[-1][:, 0], 1.5 + c * local[:, 0] - s * local[:, 1], rtol=0, atol=1e-15)
    end = args[-1]
    for start in (end.copy(), end + np.array([2 * 0.0005, 0.0, 0.0]), end + RNG.normal(0, 0.01, (20, 3))):
        _, a2 = assert_same(K.half_chord, end, start, 0.0005, np.empty(20))
        ref = np.maximum(0.5 * np.hypot(end[:, 0] - start[:, 0], end[:, 1] - start[:, 1]) - 0.0005, 0.0)
        assert np.allclose(a2[-1], ref, rtol=0, atol=1e-18)


# ----- motion -----

def test_rollout_kernel_matches_the_numpy_rollout_across_the_turn_speed_threshold():
    from robot_env import cats as CT
    V = np.array([0.0, 0.049, 0.05, 0.051, 0.3, -0.12, 1.0, 0.0, 0.0])
    W = np.array([3.0, 3.0, -3.0, 0.8, 0.0, 0.8, -3.0, 0.8, 0.0])
    L = np.array([0.0] * 7 + [0.08, -0.08])
    hold = np.array([12, 2, 2, 12, 2, 12, 2, 2, 12], dtype=np.int64)
    for v0, w0 in ((0.0, 0.0), (0.05, 2.0), (0.3, -1.0), (-0.1, 0.5)):
        dt = CT.PLAN_HORIZON / CT.PLAN_STEPS
        _, args = assert_same(K.rollout, 0.2, -0.4, 1.0, v0, w0, V, W, L, hold, dt, CT._LIMITS,
                              np.empty((9, 12)), np.empty((9, 12)), np.empty((9, 12)))
        # the replaced NumPy rollout
        x, y, yaw = np.full(9, 0.2), np.full(9, -0.4), np.full(9, 1.0)
        v, w = np.full(9, v0), np.full(9, w0)
        for k in range(12):
            held = k < hold
            v, w = CT._root_rates(v, w, V * held, W * held, dt)
            yaw = yaw + w * dt
            x = x + (v * np.cos(yaw) - L * held * np.sin(yaw)) * dt
            y = y + (v * np.sin(yaw) + L * held * np.cos(yaw)) * dt
            assert np.allclose(args[-3][:, k], x, rtol=0, atol=1e-12) and np.allclose(args[-1][:, k], yaw, rtol=0, atol=1e-12)


def test_collider_finish_including_degenerate_and_downward_capsules():
    from robot_env.cats import _quat_from_z
    n = 6
    pos = RNG.normal(0, 0.2, (n, 3))
    rot = np.tile(np.eye(3), (n, 1, 1))
    col_a, col_b = np.array([0, 1, 2, 3], dtype=np.int64), np.array([1, 1, 4, 5], dtype=np.int64)  # capsule 1: zero length
    off_a, off_b = RNG.normal(0, 0.02, (4, 3)), RNG.normal(0, 0.02, (4, 3))
    off_b[1] = off_a[1]
    pos[5] = pos[3] + off_a[3] - off_b[3] - np.array([0, 0, 0.1])  # capsule 3: pointing straight down
    _, args = assert_same(K.finish_colliders, pos, rot, col_a, off_a, col_b, off_b, 0.3, 0.4, 0.5,
                          np.empty((4, 2, 3)), np.empty((4, 4)))
    ends, quat = args[-2], args[-1]
    axis = ends[:, 1] - ends[:, 0]
    length = np.linalg.norm(axis, axis=1, keepdims=True)
    ref = _quat_from_z(np.where(length > 1e-9, axis / np.maximum(length, 1e-12), [0.0, 0.0, 1.0]))
    assert np.allclose(quat, ref, rtol=0, atol=1e-12)
    assert np.allclose(np.linalg.norm(quat, axis=1), 1.0)


def test_collider_blend_normalizes_and_hits_both_ends():
    E0, dE = RNG.normal(0, 1, (2, 5, 2, 3)), RNG.normal(0, 0.01, (2, 5, 2, 3))
    Q0 = RNG.normal(0, 1, (2, 5, 4))
    Q0 /= np.linalg.norm(Q0, axis=2, keepdims=True)
    dQ = RNG.normal(0, 0.05, (2, 5, 4))
    ids = np.arange(10, dtype=np.int64).reshape(2, 5)
    for f in (np.array([0.0, 1.0]), np.array([0.37, 0.5])):
        _, args = assert_same(K.blend_colliders, E0, dE, Q0, dQ, f, np.empty((2, 5, 2, 3)), np.zeros((10, 3)),
                              np.zeros((10, 4)), ids)
        ends, mpos, mquat = args[5], args[6], args[7]
        assert np.allclose(ends, E0 + dE * f[:, None, None, None], rtol=0, atol=1e-15)
        assert np.allclose(np.linalg.norm(mquat, axis=1), 1.0)
        assert np.allclose(mpos, (0.5 * (ends[:, :, 0] + ends[:, :, 1])).reshape(-1, 3))


def test_pose_bones_matches_mujoco_quaternions():
    import mujoco
    n = 8
    pos = RNG.normal(0, 0.2, (n, 3))
    rot = np.stack([np.linalg.qr(RNG.normal(size=(3, 3)))[0] for _ in range(n)])
    rot *= np.sign(np.linalg.det(rot))[:, None, None]
    bind = np.stack([np.linalg.qr(RNG.normal(size=(3, 3)))[0] for _ in range(n)])
    bind *= np.sign(np.linalg.det(bind))[:, None, None]
    _, args = assert_same(K.pose_bones, pos, rot, bind, 0.4, -0.2, 2.5, np.arange(n, dtype=np.int64),
                          np.zeros((n, 3)), np.zeros((n, 4)))
    c, s = math.cos(2.5), math.sin(2.5)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    for k in range(n):
        q = np.zeros(4)
        mujoco.mju_mat2Quat(q, (R @ rot[k] @ bind[k]).reshape(-1))
        got = args[-1][k]
        assert min(np.abs(got - q).max(), np.abs(got + q).max()) < 1e-12


# ----- routes -----

def test_route_kernels_match_the_plain_python_route_planner():
    from robot_env import cats as CT
    from robot_env.system import RobotSystem
    s = RobotSystem(cats=0)
    m = s.sim.model
    for t in range(150):
        start = tuple(RNG.uniform(-4.8, 4.8, 2))
        avoid = None if t % 2 else (float(RNG.uniform(-4, 4)), float(RNG.uniform(-4, 4)), 0.75)
        for _ in range(3):
            goal = tuple(RNG.uniform(-4.8, 4.8, 2))
            assert CT.plan_route(m, start, goal, avoid) == CT._plan_route_reference(m, start, goal, avoid)
    s.close()
    free = RNG.random((30, 30)) > 0.3
    free[0, 0] = True
    assert_same(K.bfs_tree, free, 0, 0, np.empty(900, dtype=np.int64))


# ----- robot -----

def test_robot_contact_flags():
    robot = np.array([True, False, False, False])
    solid = np.array([False, True, True, False])
    cat = np.array([False, False, True, False])
    for pairs, expect in (([[0, 1]], (True, False)), ([[2, 0]], (True, True)), ([[1, 3]], (False, False)),
                          (np.zeros((0, 2)), (False, False))):
        pairs = np.asarray(pairs, dtype=np.int32).reshape(-1, 2)
        assert tuple(K.robot_contacts(pairs, robot, solid, cat)) == expect
        assert_same(K.robot_contacts, pairs, robot, solid, cat)


def test_swept_safety_check_matches_the_numpy_check_including_points_at_the_buffer():
    from robot_env import safety as S
    hl, hw, buf = C.FOOTPRINT_HALF_LENGTH, C.FOOTPRINT_HALF_WIDTH, C.SAFETY_BUFFER

    def reference(poses, pts):
        X, Y, TH = poses[:, 0:1], poses[:, 1:2], poses[:, 2:3]
        dx, dy = pts[None, :, 0] - X, pts[None, :, 1] - Y
        c, s = np.cos(TH), np.sin(TH)
        dist = S.footprint_distance(c * dx + s * dy, -s * dx + c * dy)
        d0, dmin = S.footprint_distance(pts[:, 0], pts[:, 1]), dist.min(axis=0)
        out = d0 >= buf
        return bool((dmin[out] >= buf).all() and (dmin[~out] >= d0[~out] - 1e-4).all())
    edge = np.array([[hl + buf, 0.0], [0.0, hw + buf], [-(hl + buf), 0.0], [hl + buf - 1e-12, 0.0], [hl + 0.01, 0.0]])
    for t in range(3000):
        poses = S.predict_poses(RNG.uniform(-0.5, 1), RNG.uniform(-1, 1), RNG.uniform(-0.3, 1), RNG.uniform(-1.5, 1.5))
        pts = RNG.uniform(-1.2, 1.2, (int(RNG.integers(1, 40)), 2)) if t % 3 else edge[RNG.integers(0, 5, 3)]
        r, _ = assert_same(K.swept_clear, poses, pts, hl, hw, buf)
        assert bool(r) == reference(poses, pts)


# ----- the paths themselves -----

TRACE = """
import hashlib, sys
sys.path.insert(0, ROOT)
from robot_env import kernels
from robot_env.system import RobotSystem
s = RobotSystem(cats=3, cat_seed=6)
s.reset(-4.0, -2.5, 0.0, (4.0, 2.5))
h = hashlib.sha256()
for k in range(250):  # 5 s: the robot drives among the cats, turning one way then the other
    if k % 5 == 0:
        s.drive(0.5, 0.3 if (k // 60) % 2 else -0.3)
    s.advance(0.02)
    h.update(s.cats.col_world.tobytes())
    h.update(s.sim.data.qpos.tobytes())
    for c in s.cats.cats:  # what every cat decided this tick
        h.update(repr((c.state, c.state_until, c.target_v, c.target_yaw, c.cmd_v, c.cmd_w, c.cmd_lat, c.blocked,
                       c.held, c.frozen, c.escaped, sorted(c.rejected.items()), c.route, c.destination,
                       c.plan_at, c.escape, c.transitions)).encode())
print(kernels.COMPILED, h.hexdigest())
"""


def _trace(env: dict) -> str:
    code = TRACE.replace("ROOT", repr(str(ROOT)))
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=900,
                         env={**os.environ, **env}, cwd=ROOT)
    assert out.returncode == 0, out.stderr[-2000:]
    return out.stdout.strip().splitlines()[-1]


@pytest.mark.slow
def test_compiled_interpreted_and_numba_free_runs_are_identical():
    """The same 5 s run (three cats, the robot driving among them): every collider, the robot's
    state, and every cat's state, commands, guard decisions, route, and rejected commands at every
    control tick give the same digest compiled, with NUMBA_DISABLE_JIT=1, and with numba not
    installed at all (its import blocked)."""
    compiled = _trace({})
    disabled = _trace({"NUMBA_DISABLE_JIT": "1"})
    blocker = ROOT / "tests" / "_no_numba"
    blocked = _trace({"PYTHONPATH": str(blocker)})
    assert compiled.split()[0] == "True"
    assert disabled.split()[0] == "False" and blocked.split()[0] == "False"
    assert compiled.split()[1] == disabled.split()[1] == blocked.split()[1]
