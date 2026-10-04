"""Small numeric kernels for the cats (skeleton, clearance, planning), compiled with numba.

The per-sample work is many operations on small arrays (34 joints, about 20 envelope circles),
where NumPy's per-call overhead dominates; compiled, the same loops are 20-70 times faster. Each
kernel is plain Python over arrays: without numba (or with NUMBA_DISABLE_JIT=1) the same source
runs interpreted and gives bit-identical results, only slower (tests/test_kernels.py checks
both ways): the kernels use only operations that round the same way in both (so not
math.hypot, which numba computes differently). Compiled code is cached next to this file after
the first run.
"""

from __future__ import annotations

import math

import numpy as np

try:
    from numba import config as _numba_config
    from numba import njit as _njit

    COMPILED = not _numba_config.DISABLE_JIT

    def njit(f):
        return _njit(cache=True, nogil=True)(f)
except ImportError:  # the same kernels, interpreted
    COMPILED = False

    def njit(f):
        f.py_func = f
        return f


@njit
def _hypot(x, y):
    """sqrt(x^2 + y^2) (numba's math.hypot differs from Python's in the last bit; this does not)."""
    return math.sqrt(x * x + y * y)


@njit
def fk(order, parents, offsets, L, pos, rot):
    """Forward kinematics in place, joints in `order` (parents first): each joint's rotation is
    its parent's times its local rotation L, its position its parent's plus the parent's rotation
    of its bind offset. Roots (and joints not in `order`) keep the pos/rot given."""
    for idx in range(order.shape[0]):
        k = order[idx]
        p = parents[k]
        if p < 0:
            continue
        for i in range(3):
            pos[k, i] = pos[p, i] + (rot[p, i, 0] * offsets[k, 0] + rot[p, i, 1] * offsets[k, 1]
                                     + rot[p, i, 2] * offsets[k, 2])
        for i in range(3):
            for j in range(3):
                rot[k, i, j] = rot[p, i, 0] * L[k, 0, j] + rot[p, i, 1] * L[k, 1, j] + rot[p, i, 2] * L[k, 2, j]


@njit
def clearance_at(grid, origin, cell, slack, x, y):
    """Bilinear clearance from static obstacles at one point (-1 off the raster), less `slack`."""
    u = (x - origin) / cell
    v = (y - origin) / cell
    fi = math.floor(u)
    fj = math.floor(v)
    i = int(fi)
    j = int(fj)
    if i < 0 or i >= grid.shape[0] - 1 or j < 0 or j >= grid.shape[1] - 1:
        return -1.0
    fu = u - fi
    fv = v - fj
    return ((1 - fu) * (1 - fv) * grid[i, j] + fu * (1 - fv) * grid[i + 1, j]
            + (1 - fu) * fv * grid[i, j + 1] + fu * fv * grid[i + 1, j + 1]) - slack


@njit
def clearance(grid, origin, cell, slack, px, py, out):
    """clearance_at for every point of the flat arrays px, py (into out)."""
    for k in range(px.shape[0]):
        out[k] = clearance_at(grid, origin, cell, slack, px[k], py[k])


@njit
def place_circles(local, radius, x, y, yaw, out):
    """Envelope circles (x, y, r) of a cat whose circle centres in its own frame are `local`
    (n, >= 2), with its root at (x, y, yaw)."""
    c = math.cos(yaw)
    s = math.sin(yaw)
    for k in range(local.shape[0]):
        out[k, 0] = x + c * local[k, 0] - s * local[k, 1]
        out[k, 1] = y + s * local[k, 0] + c * local[k, 1]
        out[k, 2] = radius[k]


@njit
def half_chord(end, start, tol, out):
    """Half of how far each circle moves in a straight line from start to end, less tol (>= 0)."""
    for k in range(end.shape[0]):
        out[k] = max(0.5 * _hypot(end[k, 0] - start[k, 0], end[k, 1] - start[k, 1]) - tol, 0.0)


@njit
def gaps(circ, inflate, grid, origin, cell, slack, rx, ry, robot_radius, others, others_grow, use, out):
    """(wall, robot, other-cat) gaps of the circles `circ` (n, 3) grown by `inflate` (n): to static
    obstacles, to the robot's circle at (rx, ry), and to the circles of every other cat c with
    use[c] (others (cats, m, 3), each grown by others_grow (cats, m))."""
    wall = math.inf
    robot = math.inf
    other = math.inf
    for k in range(circ.shape[0]):
        r = circ[k, 2] + inflate[k]
        wall = min(wall, clearance_at(grid, origin, cell, slack, circ[k, 0], circ[k, 1]) - r)
        robot = min(robot, _hypot(circ[k, 0] - rx, circ[k, 1] - ry) - r - robot_radius)
        for c in range(others.shape[0]):
            if not use[c]:
                continue
            for q in range(others.shape[1]):
                d = _hypot(circ[k, 0] - others[c, q, 0], circ[k, 1] - others[c, q, 1])
                other = min(other, d - r - (others[c, q, 2] + others_grow[c, q]))
    out[0] = wall
    out[1] = robot
    out[2] = other


@njit
def plan_gaps(X, Y, Th, lx, ly, r, grid, origin, cell, slack, wall_gap, prx, pry, robot_term,
              ox, oy, orad, ovx, ovy, times, cat_gap, out):
    """Tightest margin (beyond each limit) of the whole cat at root poses X, Y, Th (K, S) at future
    times (S): its circles (lx, ly, r in its own frame) against static obstacles (wall_gap), the
    robot moving on to (prx, pry) at each time (its circle plus the limit: robot_term), and the
    other cats' circles (ox, oy, orad) moving on at (ovx, ovy) (cat_gap)."""
    for a in range(X.shape[0]):
        for b in range(X.shape[1]):
            c = math.cos(Th[a, b])
            s = math.sin(Th[a, b])
            t = times[b]
            tight = math.inf
            for k in range(lx.shape[0]):
                cx = X[a, b] + c * lx[k] - s * ly[k]
                cy = Y[a, b] + s * lx[k] + c * ly[k]
                tight = min(tight, clearance_at(grid, origin, cell, slack, cx, cy) - r[k] - wall_gap)
                tight = min(tight, _hypot(cx - prx[b], cy - pry[b]) - r[k] - robot_term)
                for q in range(ox.shape[0]):
                    d = _hypot(cx - (ox[q] + ovx[q] * t), cy - (oy[q] + ovy[q] * t))
                    tight = min(tight, d - r[k] - orad[q] - cat_gap)
            out[a, b] = tight


@njit
def blend_colliders(E0, dE, Q0, dQ, f, ends, mocap_pos, mocap_quat, ids):
    """Every cat's colliders at blend fraction f (per cat): endpoints lerped, orientations
    normalized-lerped; written to `ends` and to the mocap bodies `ids` (cats, colliders) as the
    capsule centre and orientation."""
    for c in range(E0.shape[0]):
        for k in range(E0.shape[1]):
            for e in range(2):
                for i in range(3):
                    ends[c, k, e, i] = E0[c, k, e, i] + dE[c, k, e, i] * f[c]
            q0 = Q0[c, k, 0] + dQ[c, k, 0] * f[c]
            q1 = Q0[c, k, 1] + dQ[c, k, 1] * f[c]
            q2 = Q0[c, k, 2] + dQ[c, k, 2] * f[c]
            q3 = Q0[c, k, 3] + dQ[c, k, 3] * f[c]
            n = math.sqrt(q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3)
            m = ids[c, k]
            mocap_quat[m, 0] = q0 / n
            mocap_quat[m, 1] = q1 / n
            mocap_quat[m, 2] = q2 / n
            mocap_quat[m, 3] = q3 / n
            for i in range(3):
                mocap_pos[m, i] = 0.5 * (ends[c, k, 0, i] + ends[c, k, 1, i])


@njit
def ieee_remainder(x, y):
    """x - n y with n the integer nearest x / y (ties to even), as math.remainder."""
    n = round(x / y)
    return x - n * y


@njit
def _rot_x(a, out):
    c, s = math.cos(a), math.sin(a)
    out[0, 0], out[0, 1], out[0, 2] = 1.0, 0.0, 0.0
    out[1, 0], out[1, 1], out[1, 2] = 0.0, c, -s
    out[2, 0], out[2, 1], out[2, 2] = 0.0, s, c


@njit
def _rot_y(a, out):
    c, s = math.cos(a), math.sin(a)
    out[0, 0], out[0, 1], out[0, 2] = c, 0.0, s
    out[1, 0], out[1, 1], out[1, 2] = 0.0, 1.0, 0.0
    out[2, 0], out[2, 1], out[2, 2] = -s, 0.0, c


@njit
def _matmul3(a, b, out):
    for i in range(3):
        for j in range(3):
            out[i, j] = a[i, 0] * b[0, j] + a[i, 1] * b[1, j] + a[i, 2] * b[2, j]


@njit
def _tmatvec3(a, v, out):
    """a transposed times v."""
    for i in range(3):
        out[i] = a[0, i] * v[0] + a[1, i] * v[1] + a[2, i] * v[2]


@njit
def leg_ik(target, pos, rot, bind_pos, parents, scap, upper, lower, paw, out):
    """Two-bone leg IK with a shoulder blade (front legs; scap < 0 for hind legs): the local
    rotations of the blade, hip or shoulder, knee or elbow, and paw (out[0..3]; out[0] is the
    identity without a blade) that put the paw joint at the target (cat frame), given the pose
    of the parents (pos, rot). The method is described at CatAnimator._leg_ik."""
    two_pi = 2.0 * math.pi
    pitch_total = 0.0
    Rp = np.empty((3, 3))
    upper_pos = np.empty(3)
    tmp = np.empty(3)
    m = np.empty((3, 3))
    for i in range(3):
        for j in range(3):
            out[0, i, j] = 1.0 if i == j else 0.0
    if scap >= 0:
        Rs_parent = rot[parents[scap]]
        for i in range(3):
            tmp[i] = target[i] - pos[scap, i]
        ts = np.empty(3)
        _tmatvec3(Rs_parent, tmp, ts)
        ax = bind_pos[upper, 0] - bind_pos[scap, 0]
        az = bind_pos[upper, 2] - bind_pos[scap, 2]
        r = _hypot(ax, az)
        dist = math.sqrt((bind_pos[paw, 0] - bind_pos[upper, 0]) ** 2 + (bind_pos[paw, 2] - bind_pos[upper, 2]) ** 2)
        lat_s = bind_pos[paw, 1] - bind_pos[scap, 1]
        t0 = ts[0]
        t1 = -_hypot(ts[2], ts[1] - lat_s)
        D = _hypot(t0, t1)
        phi0 = math.atan2(az, ax)
        if abs(r - dist) < D < r + dist:
            a_ = (r * r - dist * dist + D * D) / (2 * D)
            h = math.sqrt(max(r * r - a_ * a_, 0.0))
            u0, u1 = t0 / D, t1 / D
            p0, p1 = -u1, u0
            c1x, c1y = u0 * a_ + p0 * h, u1 * a_ + p1 * h
            c2x, c2y = u0 * a_ - p0 * h, u1 * a_ - p1 * h
            e1 = abs(ieee_remainder(math.atan2(c1y, c1x) - phi0, two_pi))
            e2 = abs(ieee_remainder(math.atan2(c2y, c2x) - phi0, two_pi))
            if e2 < e1:
                phi = math.atan2(c2y, c2x)
            else:
                phi = math.atan2(c1y, c1x)
        else:
            phi = math.atan2(t1, t0)
        sa = -ieee_remainder(phi - phi0, two_pi)
        _rot_y(sa, out[0])
        pitch_total += sa
        _matmul3(Rs_parent, out[0], Rp)
        for i in range(3):
            tmp[i] = bind_pos[upper, i] - bind_pos[scap, i]
        for i in range(3):
            upper_pos[i] = pos[scap, i] + (Rp[i, 0] * tmp[0] + Rp[i, 1] * tmp[1] + Rp[i, 2] * tmp[2])
    else:
        for i in range(3):
            upper_pos[i] = pos[upper, i]
            for j in range(3):
                Rp[i, j] = rot[parents[upper], i, j]
    for i in range(3):
        tmp[i] = target[i] - upper_pos[i]
    t = np.empty(3)
    _tmatvec3(Rp, tmp, t)
    lat0 = bind_pos[paw, 1] - bind_pos[upper, 1]
    rho = _hypot(t[1], t[2])
    delta = math.atan2(t[2], t[1])
    spread = math.acos(min(max(lat0 / max(rho, 1e-9), -1.0), 1.0))
    r1 = ieee_remainder(delta + spread, two_pi)
    r2 = ieee_remainder(delta - spread, two_pi)
    roll = r2 if abs(r2) < abs(r1) else r1
    _rot_x(roll, m)
    t_ = np.empty(3)
    _tmatvec3(m, t, t_)
    a0x, a0z = bind_pos[lower, 0] - bind_pos[upper, 0], bind_pos[lower, 2] - bind_pos[upper, 2]
    b0x, b0z = bind_pos[paw, 0] - bind_pos[lower, 0], bind_pos[paw, 2] - bind_pos[lower, 2]
    la, lb = _hypot(a0x, a0z), _hypot(b0x, b0z)
    t2x, t2z = t_[0], t_[2]
    dn = _hypot(t2x, t2z)
    d = min(max(dn, abs(la - lb) + 1e-4), (la + lb) * 0.999)
    c0x, c0z = a0x + b0x, a0z + b0z
    bend = 1.0 if ieee_remainder(math.atan2(a0z, a0x) - math.atan2(c0z, c0x), two_pi) > 0 else -1.0
    base = math.atan2(t2z, t2x)
    alpha = math.acos(min(max((la * la + d * d - lb * lb) / (2 * la * d), -1.0), 1.0))
    ua = base + bend * alpha
    kx, kz = math.cos(ua) * la, math.sin(ua) * la
    s = max(dn, 1e-9)
    ldx, ldz = (t2x / s) * d - kx, (t2z / s) * d - kz
    up = -(ua - math.atan2(a0z, a0x))
    low = -(math.atan2(ldz, ldx) - math.atan2(b0z, b0x)) - up
    ry = np.empty((3, 3))
    _rot_y(up, ry)
    _matmul3(m, ry, out[1])
    _rot_y(low, out[2])
    _rot_y(-(up + low + pitch_total), ry)
    _rot_x(-roll, m)
    _matmul3(ry, m, out[3])


@njit
def finish_colliders(pos, rot, col_a, off_a, col_b, off_b, x, y, yaw, ends, quat):
    """World endpoints (k, 2, 3) of the collider capsules of a pose (joints pos/rot in the cat
    frame, root at x, y, yaw), and each capsule's orientation (w, x, y, z): +z along its axis."""
    c, s = math.cos(yaw), math.sin(yaw)
    p = np.empty(3)
    for k in range(col_a.shape[0]):
        for e in range(2):
            j = col_a[k] if e == 0 else col_b[k]
            o = off_a[k] if e == 0 else off_b[k]
            for i in range(3):
                p[i] = pos[j, i] + (rot[j, i, 0] * o[0] + rot[j, i, 1] * o[1] + rot[j, i, 2] * o[2])
            ends[k, e, 0] = c * p[0] - s * p[1] + x
            ends[k, e, 1] = s * p[0] + c * p[1] + y
            ends[k, e, 2] = p[2]
        dx = ends[k, 1, 0] - ends[k, 0, 0]
        dy = ends[k, 1, 1] - ends[k, 0, 1]
        dz = ends[k, 1, 2] - ends[k, 0, 2]
        n = math.sqrt(dx * dx + dy * dy + dz * dz)
        if n > 1e-9:
            dx, dy, dz = dx / n, dy / n, dz / n
        else:
            dx, dy, dz = 0.0, 0.0, 1.0
        q0, q1, q2, q3 = 1.0 + min(max(dz, -1.0), 1.0), -dy, dx, 0.0  # +z onto d: (1 + z.d, z x d)
        if q0 < 1e-9:  # pointing straight down
            q0, q1, q2, q3 = 0.0, 1.0, 0.0, 0.0
        qn = math.sqrt(q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3)
        quat[k, 0], quat[k, 1], quat[k, 2], quat[k, 3] = q0 / qn, q1 / qn, q2 / qn, q3 / qn


@njit
def root_rates(v, w, cmd_v, cmd_w, dt, min_turn_speed, yaw_rate, pivot_rate, yaw_accel, accel):
    """One step of a cat's forward speed and turn rate toward a command (as cats._root_rates)."""
    rate = yaw_rate if abs(v) >= min_turn_speed else pivot_rate
    want = min(max(cmd_w, -rate), rate)
    w = w + min(max(want - w, -yaw_accel * dt), yaw_accel * dt)
    new_v = v + min(max(cmd_v - v, -accel * dt), accel * dt)
    if abs(w) > pivot_rate + 1e-9 and abs(new_v) < min(abs(v), min_turn_speed):
        sign = 1.0 if v > 0 else (-1.0 if v < 0 else 0.0)
        new_v = sign * min(abs(v), min_turn_speed)
    return new_v, w


@njit
def rollout(x0, y0, yaw0, v0, w0, V, W, L, hold, dt, limits, X, Y, Th):
    """Root poses (K, S) of each candidate command (V, W, L), held for hold[k] steps and then
    braking (a zero command), from (x0, y0, yaw0) at (v0, w0); limits = (min_turn_speed,
    yaw_rate, pivot_rate, yaw_accel, accel)."""
    for k in range(V.shape[0]):
        x, y, yaw, v, w = x0, y0, yaw0, v0, w0
        for s in range(X.shape[1]):
            held = 1.0 if s < hold[k] else 0.0
            v, w = root_rates(v, w, V[k] * held, W[k] * held, dt, limits[0], limits[1], limits[2], limits[3],
                              limits[4])
            lat = L[k] * held
            yaw = yaw + w * dt
            x = x + (v * math.cos(yaw) - lat * math.sin(yaw)) * dt
            y = y + (v * math.sin(yaw) + lat * math.cos(yaw)) * dt
            X[k, s] = x
            Y[k, s] = y
            Th[k, s] = yaw


@njit
def bfs_tree(free, ai, aj, came):
    """Breadth-first search over the 8-connected free grid from cell (ai, aj); a diagonal step
    needs both side cells free. came (flat, n*n) gets each reached cell's predecessor (flat
    index), -1 for the start, -2 for cells not reached. Neighbours are taken in a fixed order, so
    the predecessor of every reached cell is the one an early-exit search would give it."""
    n0, n1 = free.shape
    for k in range(came.shape[0]):
        came[k] = -2
    queue = np.empty(n0 * n1, dtype=np.int64)
    head, tail = 0, 0
    came[ai * n1 + aj] = -1
    queue[tail] = ai * n1 + aj
    tail += 1
    di = (1, -1, 0, 0, 1, 1, -1, -1)
    dj = (0, 0, 1, -1, 1, -1, 1, -1)
    while head < tail:
        c = queue[head]
        head += 1
        ci, cj = c // n1, c % n1
        for s in range(8):
            ni, nj = ci + di[s], cj + dj[s]
            if 0 <= ni < n0 and 0 <= nj < n1 and came[ni * n1 + nj] == -2 and free[ni, nj] and \
                    (di[s] == 0 or dj[s] == 0 or (free[ci + di[s], cj] and free[ci, cj + dj[s]])):
                came[ni * n1 + nj] = c
                queue[tail] = ni * n1 + nj
                tail += 1


@njit
def line_free(free, i0, j0, i1, j1):
    """The cells on the straight line between two cells (rounded per step) are all free."""
    k = max(abs(i1 - i0), abs(j1 - j0), 1)
    for t in range(k + 1):
        i = int(round(i0 + (i1 - i0) * t / k))
        j = int(round(j0 + (j1 - j0) * t / k))
        if not free[i, j]:
            return False
    return True


@njit
def straighten(free, ci, cj, keep):
    """Indices (into the cell path ci, cj) kept when the path is straightened: from each kept
    cell, the farthest later cell reachable in a free straight line. Returns how many (in keep)."""
    m = ci.shape[0]
    keep[0] = 0
    count = 1
    i = 0
    while i < m - 1:
        j = m - 1
        while j > i + 1 and not line_free(free, ci[i], cj[i], ci[j], cj[j]):
            j -= 1
        keep[count] = j
        count += 1
        i = j
    return count


WARM_SECONDS: float | None = None  # how long warm() took in this process (None: not yet)


@njit
def robot_contacts(pairs, robot, solid, cat):
    """Over the contacts' geom pairs (ncon, 2): (does the robot touch anything solid, does it
    touch a cat)."""
    hit_solid = False
    hit_cat = False
    for c in range(pairs.shape[0]):
        a, b = pairs[c, 0], pairs[c, 1]
        if (robot[a] and solid[b]) or (robot[b] and solid[a]):
            hit_solid = True
        if (robot[a] and cat[b]) or (robot[b] and cat[a]):
            hit_cat = True
    return hit_solid, hit_cat


@njit
def pose_bones(pos, rot, bind_rot, x, y, yaw, ids, mocap_pos, mocap_quat):
    """World poses of a cat's bone bodies (visual only: the skin follows them): positions and
    orientations (posed rotation times bind rotation, as quaternions w, x, y, z) for a pose in the
    cat frame (pos, rot) with its root at (x, y, yaw)."""
    c, s = math.cos(yaw), math.sin(yaw)
    w = np.empty((3, 3))
    for k in range(pos.shape[0]):
        m = ids[k]
        mocap_pos[m, 0] = c * pos[k, 0] - s * pos[k, 1] + x
        mocap_pos[m, 1] = s * pos[k, 0] + c * pos[k, 1] + y
        mocap_pos[m, 2] = pos[k, 2]
        for i in range(3):
            for j in range(3):
                r = rot[k, i, 0] * bind_rot[k, 0, j] + rot[k, i, 1] * bind_rot[k, 1, j] + rot[k, i, 2] * bind_rot[k, 2, j]
                w[i, j] = r
        for j in range(3):  # the root's turn about z
            a, b = w[0, j], w[1, j]
            w[0, j] = c * a - s * b
            w[1, j] = s * a + c * b
        # rotation matrix to quaternion (Shepperd's method: the largest component first)
        tr = w[0, 0] + w[1, 1] + w[2, 2]
        if tr > 0.0:
            q0 = 0.5 * math.sqrt(1.0 + tr)
            f = 0.25 / q0
            q1, q2, q3 = (w[2, 1] - w[1, 2]) * f, (w[0, 2] - w[2, 0]) * f, (w[1, 0] - w[0, 1]) * f
        elif w[0, 0] >= w[1, 1] and w[0, 0] >= w[2, 2]:
            q1 = 0.5 * math.sqrt(1.0 + w[0, 0] - w[1, 1] - w[2, 2])
            f = 0.25 / q1
            q0, q2, q3 = (w[2, 1] - w[1, 2]) * f, (w[0, 1] + w[1, 0]) * f, (w[0, 2] + w[2, 0]) * f
        elif w[1, 1] >= w[2, 2]:
            q2 = 0.5 * math.sqrt(1.0 - w[0, 0] + w[1, 1] - w[2, 2])
            f = 0.25 / q2
            q0, q1, q3 = (w[0, 2] - w[2, 0]) * f, (w[0, 1] + w[1, 0]) * f, (w[1, 2] + w[2, 1]) * f
        else:
            q3 = 0.5 * math.sqrt(1.0 - w[0, 0] - w[1, 1] + w[2, 2])
            f = 0.25 / q3
            q0, q1, q2 = (w[1, 0] - w[0, 1]) * f, (w[0, 2] + w[2, 0]) * f, (w[1, 2] + w[2, 1]) * f
        n = math.sqrt(q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3)
        mocap_quat[m, 0], mocap_quat[m, 1], mocap_quat[m, 2], mocap_quat[m, 3] = q0 / n, q1 / n, q2 / n, q3 / n


@njit
def swept_clear(poses, pts, half_length, half_width, buffer):
    """The robot's footprint at every pose (x, y, theta; robot frame at the start) keeps
    `buffer` from every point (robot frame), or, for a point already closer than that at the
    start, never comes closer than it is (to 0.1 mm)."""
    for k in range(pts.shape[0]):
        px, py = pts[k, 0], pts[k, 1]
        dx0 = max(abs(px) - half_length, 0.0)
        dy0 = max(abs(py) - half_width, 0.0)
        d0 = _hypot(dx0, dy0)
        need = buffer if d0 >= buffer else d0 - 1e-4
        for i in range(poses.shape[0]):
            ex, ey = px - poses[i, 0], py - poses[i, 1]
            c, s = math.cos(poses[i, 2]), math.sin(poses[i, 2])
            lx, ly = c * ex + s * ey, -s * ex + c * ey
            dx = max(abs(lx) - half_length, 0.0)
            dy = max(abs(ly) - half_width, 0.0)
            if _hypot(dx, dy) < need:
                return False
    return True


@njit
def _rot_zy(a, b, out):
    """rot_z(a) @ rot_y(b), written out."""
    ca, sa, cb, sb = math.cos(a), math.sin(a), math.cos(b), math.sin(b)
    out[0, 0], out[0, 1], out[0, 2] = ca * cb, -sa, ca * sb
    out[1, 0], out[1, 1], out[1, 2] = sa * cb, ca, sa * sb
    out[2, 0], out[2, 1], out[2, 2] = -sb, 0.0, cb


@njit
def posture(L, spine, bend, breathe, neck, head, head_yaw, tail, tail_rate, time, tail_phase, amp, tail_lift):
    """The local rotations of the spine (bent into the turn, breathing at the second joint), the
    neck and head (turned by head_yaw), and the tail (a wave running to the tip, raised by
    tail_lift, mostly at the base), written into L (see CatAnimator.update)."""
    for k in range(spine.shape[0]):
        _rot_zy(bend / spine.shape[0], breathe if k == 1 else 0.0, L[spine[k]])
    _rot_zy(0.4 * head_yaw, 0.0, L[neck])
    _rot_zy(0.6 * head_yaw, 0.0, L[head])
    for k in range(tail.shape[0]):
        wave = math.sin(2 * math.pi * tail_rate * time + tail_phase - 0.7 * k)
        lift = tail_lift * (0.6 if k == 0 else 0.2 if k == 1 else 0.05)
        _rot_zy(amp * wave * (0.5 + 0.15 * k), lift, L[tail[k]])


def warm() -> float:
    """Compile (or load from the cache) every kernel now, once per process, so the first frames
    do not pay for it. Returns the seconds the first call took."""
    global WARM_SECONDS
    if WARM_SECONDS is not None:
        return WARM_SECONDS
    import time
    t = time.perf_counter()
    z3, z1 = np.zeros((1, 3)), np.zeros(1)
    grid = np.zeros((3, 3))
    fk(np.zeros(1, dtype=np.int64), np.full(1, -1, dtype=np.int64), z3, np.zeros((1, 3, 3)), z3.copy(),
       np.zeros((1, 3, 3)))
    clearance(grid, 0.0, 1.0, 0.0, z1, z1, z1.copy())
    place_circles(z3, z1, 0.0, 0.0, 0.0, z3.copy())
    half_chord(z3, z3, 0.0, z1.copy())
    gaps(z3, z1, grid, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, np.zeros((1, 1, 3)), np.zeros((1, 1)),
         np.zeros(1, dtype=np.bool_), np.zeros(3))
    z2 = np.zeros((1, 1))
    plan_gaps(z2, z2, z2, z1, z1, z1, grid, 0.0, 1.0, 0.0, 0.0, z1, z1, 0.0, z1, z1, z1, z1, z1, z1, 0.0, z2.copy())
    blend_colliders(np.zeros((1, 1, 2, 3)), np.zeros((1, 1, 2, 3)), np.ones((1, 1, 4)), np.zeros((1, 1, 4)), z1,
                    np.zeros((1, 1, 2, 3)), z3.copy(), np.zeros((1, 4)), np.zeros((1, 1), dtype=np.int64))
    bind = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, -0.1], [0.02, 0.0, -0.2]])  # a two-bone leg
    leg_ik(np.array([0.0, 0.0, -0.18]), bind.copy(), np.tile(np.eye(3), (3, 1, 1)), bind,
           np.array([-1, 0, 1], dtype=np.int64), -1, 0, 1, 2, np.zeros((4, 3, 3)))
    finish_colliders(z3, np.zeros((1, 3, 3)), np.zeros(1, dtype=np.int64), z3, np.zeros(1, dtype=np.int64), z3,
                     0.0, 0.0, 0.0, np.zeros((1, 2, 3)), np.zeros((1, 4)))
    rollout(0.0, 0.0, 0.0, 0.0, 0.0, z1, z1, z1, np.zeros(1, dtype=np.int64), 0.1, np.ones(5), z2.copy(), z2.copy(),
            z2.copy())
    free = np.ones((3, 3), dtype=np.bool_)
    came = np.empty(9, dtype=np.int64)
    bfs_tree(free, 0, 0, came)
    straighten(free, np.zeros(2, dtype=np.int64), np.ones(2, dtype=np.int64), np.empty(2, dtype=np.int64))
    pose_bones(z3, np.ones((1, 3, 3)), np.ones((1, 3, 3)), 0.0, 0.0, 0.0, np.zeros(1, dtype=np.int64), z3.copy(),
               np.zeros((1, 4)))
    swept_clear(np.zeros((1, 3)), np.ones((1, 2)), 0.1, 0.1, 0.05)
    one = np.zeros(1, dtype=np.int64)
    posture(np.zeros((1, 3, 3)), one, 0.0, 0.0, 0, 0, 0.0, one, 0.0, 0.0, 0.0, 0.0, 0.0)
    flags = np.zeros(1, dtype=np.bool_)
    robot_contacts(np.zeros((1, 2), dtype=np.int32), flags, flags, flags)
    WARM_SECONDS = time.perf_counter() - t
    return WARM_SECONDS
