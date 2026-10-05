"""What changed in the physical world between a base revision and the working tree (development
check for furniture changes).

    .venv\\Scripts\\python tools\\world_diff.py [--base REV] [--seeds 1000]

Both worlds are compiled the way the simulator does it (the base world.xml from git, the current
one from robot_env/world.xml) and compared:

- every solid world geom (not the robot): removed, added, or changed, field by field (type, world
  position and yaw, size, contact type and affinity, condim, friction, solref, solimp, margin, group);
- the planner's inflated occupancy grid: cells newly blocked or freed, connected free regions,
  and an image (qa_output/furniture/world_diff.png: red newly blocked, green newly free);
- lidar scans at fixed poses (largest change, rays changed by more than 1 cm);
- each furniture object's footprint (its solids grouped by name) and how far it moved;
- the agreed route gates on the current world: no furniture solid in a doorway zone (the 0.9 m
  aperture plus R on both sides, 0.65 m into each room), a free-grid crossing at every doorway, a
  corridor route, the corridor's free span and least clearance not reduced, and, cell by cell,
  every cell that is a start/goal candidate in both worlds and in the base's main free region
  still in the main free region, with no new region holding candidate cells (the base world
  already has one isolated region, the storage room's west aisle); newly blocked candidate cells
  are listed by the nearest object, new pockets with their area. (Every free cell of the inflated
  grid has clearance >= R, so the robot can turn on the spot anywhere it can be.)
- separately: the first N task seeds sampled (sample_task) all reachable, and the worst estimated
  travel time (at most 65 s). That is about the sampled tasks, not about connectivity.

Writes qa_output/furniture/world_diff.json; exit status non-zero if a route gate fails.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import mujoco
import numpy as np
from scipy import ndimage

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from robot_env import config as C  # noqa: E402
from robot_env.layout import RoomMap  # noqa: E402
from robot_env.sim import RobotSim, load_assets  # noqa: E402

OUT = ROOT / "qa_output" / "furniture"
R = C.CIRCUMSCRIBED_RADIUS + C.PLANNING_CLEARANCE
THROAT = 2 * R + C.GRID_RESOLUTION
WALL_HALF = 0.05
DOORS = [("office", -2.5, 0.75, "x"), ("lab", 2.5, 0.75, "x"), ("storage", -3.0, -0.75, "x"),
         ("reception", 2.0, -0.75, "x"), ("office_lab", 0.0, 3.5, "y")]
SCAN_POSES = [(-3.8, 3.0, math.pi / 2), (-4.0, 2.4, math.pi), (-2.5, 2.0, 0.0), (-1.0, 3.0, math.pi / 2),
              (-0.45, 2.0, math.pi / 2), (-2.5, 1.2, math.pi / 2), (2.0, 2.0, 0.0), (-2.0, -2.0, 0.0),
              (2.5, -3.0, -math.pi / 2), (0.0, 0.0, 0.0)]
OLD_PREFIX = {"office_desk": "office_desk", "office_chair": "office_chair", "office_bookshelf": "office_bookshelf",
              "office_cabinet": "office_cabinet", "office_plant": "office_plant", "office_trash_bin": "office_trash_bin",
              "office_lamp": "office_lamp"}


def compile_world(xml: str) -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_string(xml, load_assets())


def solids(model: mujoco.MjModel) -> dict[str, dict]:
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    robot = model.body("robot").id
    out = {}
    for g in range(model.ngeom):
        if model.geom_contype[g] == 0 or model.body_rootid[model.geom_bodyid[g]] == robot:
            continue
        name = model.geom(g).name or f"#{g}"
        xm = data.geom_xmat[g].reshape(3, 3)
        out[name] = {"type": int(model.geom_type[g]), "pos": np.round(data.geom_xpos[g], 5).tolist(),
                     "yaw": round(math.degrees(math.atan2(xm[1, 0], xm[0, 0])), 4),
                     "size": np.round(model.geom_size[g], 5).tolist(), "contype": int(model.geom_contype[g]),
                     "conaffinity": int(model.geom_conaffinity[g]), "condim": int(model.geom_condim[g]),
                     "friction": np.round(model.geom_friction[g], 5).tolist(),
                     "solref": np.round(model.geom_solref[g], 5).tolist(),
                     "solimp": np.round(model.geom_solimp[g], 5).tolist(),
                     "margin": float(model.geom_margin[g]), "group": int(model.geom_group[g])}
    return out


def footprint(s: dict) -> tuple[float, float, float, float]:
    x, y, _ = s["pos"]
    if s["type"] == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
        r = s["size"][0]
        return x - r, x + r, y - r, y + r
    c, sn = abs(math.cos(math.radians(s["yaw"]))), abs(math.sin(math.radians(s["yaw"])))
    hx = c * s["size"][0] + sn * s["size"][1]
    hy = sn * s["size"][0] + c * s["size"][1]
    return x - hx, x + hx, y - hy, y + hy


def objects(sol: dict, prefixes: dict | None = None) -> dict[str, list]:
    """Each furniture object's plan bounding box: its solids grouped by name prefix (an item name of
    tools/build_world.py; the old procedural shapes used the same names)."""
    out = {}
    for name, s in sol.items():
        for prefix, obj in (prefixes or OLD_PREFIX).items():
            if name == prefix or name.startswith(prefix + "_"):
                b = footprint(s)
                o = out.setdefault(obj, [math.inf, -math.inf, math.inf, -math.inf])
                out[obj] = [min(o[0], b[0]), max(o[1], b[1]), min(o[2], b[2]), max(o[3], b[3])]
    return {k: [round(v, 4) for v in b] for k, b in out.items()}


def route_gates(model: mujoco.MjModel, room: RoomMap, base_room: RoomMap, sol: dict, items: set) -> dict:
    res = {}
    zones = []
    for name, x, y, axis in DOORS:
        half_across = 0.45 + R
        if axis == "x":
            zones.append((name, x - half_across, x + half_across, y - WALL_HALF - 0.65, y + WALL_HALF + 0.65))
        else:
            zones.append((name, x - WALL_HALF - 0.65, x + WALL_HALF + 0.65, y - half_across, y + half_across))
    intrusions = []
    for gname, s in sol.items():
        if not any(gname == i or gname.startswith(i + "_") for i in items):
            continue
        b = footprint(s)
        for zname, x0, x1, y0, y1 in zones:
            if b[0] < x1 and b[1] > x0 and b[2] < y1 and b[3] > y0:
                intrusions.append((gname, zname))
    res["doorway_zone_intrusions"] = intrusions
    crossings = {}
    for name, x, y, axis in DOORS:
        a, b = ((x, y - 0.8), (x, y + 0.8)) if axis == "x" else ((x - 0.8, y), (x + 0.8, y))
        crossings[name] = room.find_path(a, b) is not None
    res["doorway_crossings"] = crossings
    ends = [x for x in np.arange(-4.9, 4.95, 0.05) if room.free[room.cell(x, 0.0)]]
    res["corridor_route"] = bool(ends) and room.find_path((ends[0], 0.0), (ends[-1], 0.0)) is not None
    base_ends = [x for x in np.arange(-4.9, 4.95, 0.05) if base_room.free[base_room.cell(x, 0.0)]]
    res["corridor_free_span"] = [round(ends[0], 2), round(ends[-1], 2)] if ends else None
    res["corridor_free_span_base"] = [round(base_ends[0], 2), round(base_ends[-1], 2)] if base_ends else None
    xs = np.linspace(-4.6, 4.6, 400)
    res["corridor_min_clearance"] = round(float(np.min([room.point_clearance(x, 0.0) for x in xs])), 4)
    res["corridor_min_clearance_base"] = round(float(np.min([base_room.point_clearance(x, 0.0) for x in xs])), 4)
    def labelled(rm):
        lab, n = ndimage.label(rm.free)
        eligible = rm.free & (rm.clear >= C.START_GOAL_MIN_CLEARANCE)
        comps, counts = np.unique(lab[eligible], return_counts=True)
        main = int(comps[np.argmax(counts)]) if len(comps) else 0
        return lab, eligible, main, {int(c): int(k) for c, k in zip(comps, counts) if c}
    lab_b, el_b, main_b, comps_b = labelled(base_room)
    lab_n, el_n, main_n, comps_n = labelled(room)
    both_main = el_b & el_n & (lab_b == main_b)
    left_main = both_main & (lab_n != main_n)
    new_regions = []
    for c, count in comps_n.items():
        if c == main_n:
            continue
        overlap = set(np.unique(lab_b[(lab_n == c) & el_b]).tolist()) - {0}
        if not (overlap & (set(comps_b) - {main_b})):
            new_regions.append({"cells": count})
    res["candidate_cells_left_main_region"] = int(left_main.sum())
    res["new_regions_with_candidate_cells"] = new_regions
    res["candidate_regions_base"] = sorted(comps_b.values())
    res["candidate_regions_now"] = sorted(comps_n.values())
    lost = el_b & ~el_n
    centres = {k: ((b[0] + b[1]) / 2, (b[2] + b[3]) / 2) for k, b in objects(sol, {i: i for i in items}).items()}
    by_object = {}
    for i, j in np.argwhere(lost):
        x, y = float(room.cx[i, j]), float(room.cy[i, j])
        k = min(centres, key=lambda o: math.dist(centres[o], (x, y))) if centres else "?"
        by_object[k] = by_object.get(k, 0) + 1
    res["candidate_cells_newly_blocked_by_nearest_object"] = by_object
    pockets = []
    for c in range(1, int(lab_n.max()) + 1):
        cells = lab_n == c
        if c == main_n or (cells & el_n).any():
            continue
        if (np.unique(lab_b[cells]) > 0).all() and len(set(np.unique(lab_b[cells]).tolist())) == 1                 and (lab_b == np.unique(lab_b[cells])[0]).sum() == cells.sum():
            continue  # the same pocket existed before
        ij = np.argwhere(cells)
        pockets.append({"cells": int(len(ij)), "centre": [round(float(room.cx[tuple(ij.mean(0).astype(int))]), 2),
                                                          round(float(room.cy[tuple(ij.mean(0).astype(int))]), 2)],
                        "connected_to_main": False, "turn_circle": "every free cell (clearance >= R)"})
    res["new_pockets"] = pockets
    res["pass"] = (not intrusions and all(crossings.values()) and res["corridor_route"]
                   and res["candidate_cells_left_main_region"] == 0 and not new_regions
                   and res["corridor_min_clearance"] >= res["corridor_min_clearance_base"] - 1e-9
                   and res["corridor_free_span"] == res["corridor_free_span_base"])
    return res


def seed_sweep(room: RoomMap, n: int) -> dict:
    worst, worst_seed, failed = 0.0, None, []
    for seed in range(n):
        try:
            t = room.sample_task(seed)
        except Exception as e:  # noqa: BLE001 (record and continue)
            failed.append((seed, str(e)))
            continue
        tt = t.estimated_travel_time()
        if tt > worst:
            worst, worst_seed = tt, seed
    return {"sampled_tasks": n, "unreachable": failed, "worst_travel_s": round(worst, 2), "worst_seed": worst_seed,
            "pass": not failed and worst <= 65.0}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="HEAD")
    ap.add_argument("--seeds", type=int, default=1000)
    args = ap.parse_args()
    base_xml = subprocess.run(["git", "show", f"{args.base}:robot_env/world.xml"], cwd=ROOT, capture_output=True,
                              text=True, check=True, encoding="utf-8").stdout
    new_xml = (ROOT / "robot_env" / "world.xml").read_text(encoding="utf-8")
    base_m, new_m = compile_world(base_xml), compile_world(new_xml)
    sb, sn = solids(base_m), solids(new_m)
    removed = sorted(set(sb) - set(sn))
    added = sorted(set(sn) - set(sb))
    changed = {}
    for name in sorted(set(sb) & set(sn)):
        diff = {k: [sb[name][k], sn[name][k]] for k in sb[name] if sb[name][k] != sn[name][k]}
        if diff:
            changed[name] = diff
    base_room, room = RoomMap(base_m), RoomMap(new_m)
    blocked = base_room.free & ~room.free
    freed = ~base_room.free & room.free
    img = np.full(room.free.shape + (3,), 255, np.uint8)
    img[~room.free & ~blocked] = (90, 90, 90)
    img[blocked] = (220, 40, 40)
    img[freed] = (40, 170, 60)
    OUT.mkdir(parents=True, exist_ok=True)
    from PIL import Image
    Image.fromarray(np.flipud(img.transpose(1, 0, 2))).resize((800, 800), Image.NEAREST).save(OUT / "world_diff.png")
    sys.path.insert(0, str(ROOT / "tools"))
    import build_world
    build_world.generate()
    prefixes = {it["name"]: it["name"] for it in build_world.furniture_items}
    ob, on = objects(sb, prefixes), objects(sn, prefixes)
    moves = {}
    for k in sorted(set(ob) | set(on)):
        if k in ob and k in on:
            cb = ((ob[k][0] + ob[k][1]) / 2, (ob[k][2] + ob[k][3]) / 2)
            cn = ((on[k][0] + on[k][1]) / 2, (on[k][2] + on[k][3]) / 2)
            moves[k] = {"before": ob[k], "after": on[k], "centre_moved_m": round(math.dist(cb, cn), 4)}
        else:
            moves[k] = {"before": ob.get(k), "after": on.get(k)}
    groups = np.array([1, 1, 0, 0, 1, 1], dtype=np.uint8)
    probe = RobotSim()
    probe.reset(0.0, 0.0, 0.0, (2.0, 2.0))
    lidar_z = float(probe.data.site_xpos[probe.model.site("lidar_site").id][2])  # the real plane (0.1355 m)
    probe.close()
    scan_diff = []
    for pose in SCAN_POSES:
        out = []
        for model in (base_m, new_m):
            data = mujoco.MjData(model)
            mujoco.mj_forward(model, data)
            ang = pose[2] + np.radians(np.arange(0.0, 360.0, 1.0))
            dirs = np.column_stack([np.cos(ang), np.sin(ang), np.zeros_like(ang)])
            dist = np.zeros(len(dirs))
            gid = np.zeros(len(dirs), dtype=np.int32)
            mujoco.mj_multiRay(model, data, np.array([pose[0], pose[1], lidar_z]), dirs.ravel(), groups, 1,
                               model.body("robot").id, gid, dist, None, len(dirs), C.LIDAR_RANGE)
            out.append(np.where(dist < 0, C.LIDAR_RANGE, dist))
        d = np.abs(out[0] - out[1])
        scan_diff.append({"pose": [round(p, 3) for p in pose], "largest_change_m": round(float(d.max()), 4),
                          "rays_changed_over_1cm": int(np.sum(d > 0.01))})
    sys.path.insert(0, str(ROOT / "tools"))
    import build_world
    build_world.generate()
    items = {it["name"] for it in build_world.furniture_items}
    gates = route_gates(new_m, room, base_room, sn, items)
    sweep = seed_sweep(room, args.seeds)
    base_sweep = seed_sweep(base_room, args.seeds)
    lab_b, nb = ndimage.label(base_room.free)
    lab_n, nn = ndimage.label(room.free)
    report = {"base": args.base, "solids_removed": removed, "solids_added": added, "solids_changed": changed,
              "occupancy": {"cells_newly_blocked": int(blocked.sum()), "cells_newly_free": int(freed.sum()),
                            "free_regions_base": int(nb), "free_regions_now": int(nn)},
              "objects": moves, "lidar": scan_diff, "route_gates": gates, "seed_sweep": sweep,
              "seed_sweep_base": base_sweep}
    (OUT / "world_diff.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("occupancy", "route_gates", "seed_sweep", "seed_sweep_base")}, indent=1))
    print("objects:", json.dumps(moves))
    print(f"solids: {len(removed)} removed, {len(added)} added, {len(changed)} changed")
    return 0 if gates["pass"] and sweep["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
