"""Measure the cat gait: planted-paw slip and height error, floor penetration, joint continuity.
Development check (no rendering). Floor penetration is measured on the surfaces themselves: the
lowest skinned vertex of the visible cat (linear blend skinning, as drawn) and the lowest point of
any collider capsule, in world coordinates, at every animation update.

    .venv\\Scripts\\python tools\\gait_check.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from robot_env.cat_motion import CatAnimator  # noqa: E402
from robot_env.cat_rig import load_rig  # noqa: E402
from robot_env.cats import ANIM_PERIOD, PIVOT_RATE, V_MAX, YAW_RATE, colliders  # noqa: E402

DT = 0.0005  # physics step


def run(profile, seconds: float, seed: int = 0, step: float = DT, rate_every: int = 1):
    """profile(t) -> (v, w[, lat[, sit]]). Integrates the cat's root, animates every `rate_every` steps, and
    records each paw's world position and planted flag per animation update."""
    rig = load_rig()
    an = CatAnimator(rig, seed)
    x = y = yaw = 0.0
    an.reset(x, y, yaw)
    rec = {name: [] for name in an.legs}
    paw_ids = {name: leg.contact for name, leg in an.legs.items()}
    prev_local = None
    jumps = 0.0
    # skinning: each vertex follows its joints (pos_j + rot_j (v - bind_pos_j)), weighted
    V, J, W = rig.vertices, rig.joints, rig.weights
    rel = V[:, None, :] - rig.bind_pos[J]  # (n, 4, 3)
    cols = colliders()
    ca, cb = np.array([c.bone_a for c in cols]), np.array([c.bone_b for c in cols])
    oa, ob = np.array([c.off_a for c in cols]), np.array([c.off_b for c in cols])
    radius = np.array([c.radius for c in cols])
    low = {"skin": math.inf, "colliders": math.inf}
    n = int(round(seconds / step))
    for i in range(n):
        t = i * step
        v, w, *side = profile(t)
        lat = side[0] if side else 0.0  # a side step (m/s, to the left), as the cats' planner uses
        an.sit_target = side[1] if len(side) > 1 else 0.0  # 1: sit down (the herd asks only when still)
        yaw += w * step
        x += (v * math.cos(yaw) - lat * math.sin(yaw)) * step
        y += (v * math.sin(yaw) + lat * math.cos(yaw)) * step
        if i % rate_every:
            continue
        local, bob = an.update(step * rate_every, x, y, yaw, v, w, None, lat)
        pos, rot = an._fk(local)
        pos[:, 0] += an.shift  # sitting: the body moves forward
        # heights do not depend on the root's x, y, yaw: the cat frame's z plus the body bob
        skin_z = (W * (pos[J][:, :, 2] + np.einsum("nkj,nkj->nk", rot[J][:, :, 2, :], rel))).sum(1) + bob
        za = pos[ca][:, 2] + np.einsum("nj,nj->n", rot[ca][:, 2, :], oa) + bob
        zb = pos[cb][:, 2] + np.einsum("nj,nj->n", rot[cb][:, 2, :], ob) + bob
        low["skin"] = min(low["skin"], float(skin_z.min()))
        low["colliders"] = min(low["colliders"], float((np.minimum(za, zb) - radius).min()))
        c, s = math.cos(yaw), math.sin(yaw)
        for name, leg in an.legs.items():
            p = pos[paw_ids[name]]
            world = np.array([x + c * p[0] - s * p[1], y + s * p[0] + c * p[1], p[2] + bob])
            rec[name].append((t, world, leg.planted))
        if prev_local is not None:
            for j, R in local.items():
                if j in prev_local:
                    jumps = max(jumps, float(np.degrees(np.arccos(np.clip((np.trace(prev_local[j].T @ R) - 1) / 2, -1, 1)))))
        prev_local = local
    return rec, jumps, low


def metrics(rec, blend: float = 0.05):
    """Max planted-paw horizontal speed and height error (excluding `blend` s after touchdown
    and before lift-off), measured at the paw joints."""
    worst_v, worst_z = 0.0, 0.0
    for name, samples in rec.items():
        ts = np.array([s[0] for s in samples])
        ps = np.array([s[1] for s in samples])
        pl = np.array([s[2] for s in samples])
        ground = ps[pl, 2].min() if pl.any() else 0.0
        # planted runs
        i = 0
        while i < len(pl):
            if not pl[i]:
                i += 1
                continue
            j = i
            while j < len(pl) and pl[j]:
                j += 1
            seg = slice(i, j)
            tt, pp = ts[seg], ps[seg]
            keep = (tt >= tt[0] + blend) & (tt <= tt[-1] - blend)
            if keep.sum() > 2:
                vel = np.linalg.norm(np.diff(pp[keep][:, :2], axis=0), axis=1) / np.diff(tt[keep])
                worst_v = max(worst_v, float(vel.max()))
                worst_z = max(worst_z, float(np.abs(pp[keep][:, 2] - ground).max()))
            i = j
    return {"stance_slip_mps": round(worst_v, 4), "stance_z_err_m": round(worst_z, 4)}


PROFILES = {
    "walk 0.3": lambda t: (0.3, 0.0),
    "walk 0.2 turning": lambda t: (0.2, 0.6),
    "turn in place": lambda t: (0.0, PIVOT_RATE),  # the fastest the cats pivot
    "pivot 0.6": lambda t: (0.0, 0.6),
    "start-stop": lambda t: ((0.25 if (t % 3) < 1.8 else 0.0), 0.0),
    "trot 0.9": lambda t: (0.9, 0.0),
    "run 1.0": lambda t: (V_MAX, 0.0),  # the fastest a cat goes (a yield hurrying from a fast robot)
    "run 1.0 turning": lambda t: (V_MAX, YAW_RATE),  # ... turning as hard as the planner allows
    "run 1.0 swerving": lambda t: (V_MAX, (YAW_RATE if (t % 1.0) < 0.5 else -YAW_RATE)),
    "back up 0.12": lambda t: (-0.12, 0.0),
    "back up turning": lambda t: (-0.12, 0.8),
    "side step 0.08": lambda t: (0.0, 0.0, 0.08),
    "side step back": lambda t: (0.0, 0.0, (0.08 if (t % 2) < 1 else -0.08)),
    # walk, stop, sit down, get up, walk on (moving again only once up, as the herd does)
    "sit and stand": lambda t: ((0.2 if t < 0.8 or t >= 3.4 else 0.0), 0.0, 0.0, (1.0 if 1.0 <= t < 2.8 else 0.0)),
}

SLIP_MAX = 0.01  # m/s: a planted paw (outside 50 ms of touchdown and lift-off)
HEIGHT_MAX = 0.002  # m: a planted paw's height error
FLOOR_MAX = 0.001  # m: how far below the floor the visible cat (skin) or a collider may go
JOINT_STEP_MAX = 25.0  # degrees per 10 ms sample: a knee or elbow flipping its bend jumps 90+


def main() -> int:
    ok = True
    for name, prof in PROFILES.items():
        rec, jumps, low = run(prof, 4.0, rate_every=round(ANIM_PERIOD / DT))  # the cats' own 100 Hz
        m = metrics(rec)
        m["lowest_skin_m"], m["lowest_collider_m"] = round(low["skin"], 5), round(low["colliders"], 5)
        passed = (m["stance_slip_mps"] <= SLIP_MAX and m["stance_z_err_m"] <= HEIGHT_MAX
                  and min(low.values()) >= -FLOOR_MAX and jumps <= JOINT_STEP_MAX)
        ok &= passed
        print(f"{name:18s}", m, "max joint step deg", round(jumps, 2), "PASS" if passed else "FAIL")
    print("gait gate:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
