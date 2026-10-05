"""Do the furniture meshes and their solid members agree, and can the lidar see everything the
robot can touch? (development check and test helper)

    .venv\\Scripts\\python tools\\furniture_check.py [item names]

Per furniture item of the world (tools/build_world.py item()), at every height from the floor to
the gate height (max(0.40 m, the robot's colliding top + 0.05 m)) in 5 mm steps, plus 1 mm below
and above every member's bottom and top, the mesh's filled section and the members' section are
rasterized (5 mm cells; a cell's centre can be up to RASTER_M = 3.5 mm from the true surface, so
every limit below is tightened by it):

- protrusion: no mesh cell lies outside the members grown by 1 cm (less the raster term): no
  visible part where the robot would pass through;
- member surfaces: every exposed member cell (a member cell next to one outside all members) is
  within 2 cm of the visible mesh, measured + RASTER_M (target 5 mm; counts over 5 and 10 mm per
  member). Every horizontal sight line's first member hit is such a cell, so a sensor or a
  collision never meets a member more than 2 cm in front of something visible;
- sight lines (reported, not gated): along dense horizontal lines from every direction (1 degree),
  how far the first mesh hit lies beyond the first member hit (a grazing line can miss a thin part
  and meet a far one, so this is what a viewer sees, not a geometric bound);
- open frames: where the members are solid but the mesh is open from the floor to the robot's
  top, the largest inscribed radius of that region (+ RASTER_M) stays below 0.14 m, half the
  footprint's short side (a 0.28 x 0.32 m footprint in any orientation contains such a disc);
- support: every mesh's base stands on the floor or on the top of one of the item's members
  (3 mm).

And over the whole compiled world (lidar_gate): every solid static geom that reaches below the
robot's colliding top either spans the lidar plane with 1 cm to spare above and below, or lies,
seen from above, inside the solids that do, grown by 2 cm. (The lidar is a single plane: what is
wholly below it is invisible to it, so the robot must always meet a seen solid first.)
And overlap_gate: at every 1 cm of height, no mesh's filled section passes into another mesh (its
own item's too), a wall, a door, or another item's solids, beyond contact.

Results go to qa_output/furniture/check.json; the exit status is non-zero if any gate fails.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))
import build_world  # noqa: E402
import furniture_geom as fg  # noqa: E402

RES = 0.005
RASTER_M = RES * math.sqrt(2) / 2  # a cell centre to the farthest point of the cell
PROTRUSION_M = 0.01
SURFACE_M = 0.02
SURFACE_TARGET_M = 0.005
FOOTPRINT_HALF_MIN = 0.14  # half the shorter side of the robot's footprint (0.28 x 0.32 m)
SUPPORT_M = 0.003
PLANE_MARGIN_M = 0.01
COVER_GROW_M = 0.02
OUT = ROOT / "qa_output" / "furniture" / "check.json"


def compiled_world():
    """The simulator's world (robot_env/world.xml, which a test keeps equal to the generator's
    output) with the robot at its spawn pose; build_world's registry is filled as well."""
    from robot_env.sim import RobotSim
    build_world.generate()
    sim = RobotSim()
    sim.reset(0.0, 0.0, 0.0, (2.0, 2.0))
    return sim


def geom_z_extent(model, data, g) -> tuple[float, float]:
    """Exact lowest and highest z of a geom (box, cylinder, capsule, sphere, ellipsoid), turned."""
    import mujoco
    t = model.geom_type[g]
    size = model.geom_size[g]
    r = data.geom_xmat[g].reshape(3, 3)
    z = float(data.geom_xpos[g][2])
    if t == mujoco.mjtGeom.mjGEOM_BOX:
        h = float(np.abs(r[2]) @ size)
    elif t == mujoco.mjtGeom.mjGEOM_CYLINDER:
        h = abs(r[2, 2]) * size[1] + math.sqrt(max(0.0, 1 - r[2, 2] ** 2)) * size[0]
    elif t == mujoco.mjtGeom.mjGEOM_CAPSULE:
        h = abs(r[2, 2]) * size[1] + size[0]
    elif t == mujoco.mjtGeom.mjGEOM_SPHERE:
        h = size[0]
    elif t == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
        h = float(np.sqrt(((r[2] * size) ** 2).sum()))
    else:
        raise ValueError(f"geom {model.geom(g).name}: no exact extent for type {t}")
    return z - h, z + h


def robot_top_and_plane(model, data) -> tuple[float, float]:
    """The robot's colliding top at its spawn pose (exact, per geom type) and the lidar plane."""
    robot = model.body("robot").id
    top = max(geom_z_extent(model, data, g)[1] for g in range(model.ngeom)
              if model.body_rootid[model.geom_bodyid[g]] == robot and model.geom_contype[g] != 0)
    return float(top), float(data.site_xpos[model.site("lidar_site").id][2])


def member_shapes(model, data, names):
    """(name, kind, centre xyz, yaw, sizes) of the named geoms in world coordinates."""
    import mujoco
    out = []
    for n in names:
        g = model.geom(n).id
        xmat = data.geom_xmat[g].reshape(3, 3)
        if abs(xmat[2, 2] - 1.0) > 1e-9:
            raise ValueError(f"member {n} is not upright")
        kind = "box" if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX else "cyl"
        out.append((n, kind, data.geom_xpos[g].copy(), math.atan2(xmat[1, 0], xmat[0, 0]), model.geom_size[g].copy()))
    return out


def _half_height(kind, size):
    return size[2] if kind == "box" else size[1]


def member_raster(shapes, z, lo, shape, grow=0.0, label=False):
    """Members' section at height z (optionally grown by `grow` metres); with label=True, which
    member (index + 1) covers each cell (the first one listed)."""
    ix, iy = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing="ij")
    px, py = lo[0] + (ix + 0.5) * RES, lo[1] + (iy + 0.5) * RES
    occ = np.zeros(shape, np.int32 if label else bool)
    for k, (_, kind, c, yaw, size) in enumerate(shapes):
        hz = _half_height(kind, size)
        if not (c[2] - hz - grow - 1e-9 <= z <= c[2] + hz + grow + 1e-9):  # stacked members share faces
            continue
        dx, dy = px - c[0], py - c[1]
        if kind == "box":
            cs, sn = math.cos(yaw), math.sin(yaw)
            lx, ly = cs * dx + sn * dy, -sn * dx + cs * dy
            inside = (np.abs(lx) <= size[0] + grow) & (np.abs(ly) <= size[1] + grow)
        else:
            inside = np.hypot(dx, dy) <= size[0] + grow
        if label:
            occ[inside & (occ == 0)] = k + 1
        else:
            occ |= inside
    return occ


def surface_cells(mesh: np.ndarray, memb: np.ndarray):
    """Exposed member cells and their distance (metres, cell centres) to the nearest mesh cell."""
    edge = memb & ~ndimage.binary_erosion(memb, structure=np.ones((3, 3), bool), border_value=0)
    cells = np.argwhere(edge)
    if not mesh.any():
        return cells, np.full(len(cells), np.inf)
    return cells, (ndimage.distance_transform_edt(~mesh) * RES)[edge]


def sight_line_excess(mesh: np.ndarray, memb: np.ndarray, step_deg: float = 1.0):
    """For dense horizontal sight lines (RES apart, every step_deg) that meet both: how far the
    first mesh hit lies beyond the first member hit (metres; negative: the mesh is seen first),
    and the cell (ix, iy) where the worst line first meets a member."""
    nx, ny = mesh.shape
    d = int(math.ceil(math.hypot(nx, ny))) + 2
    c_in = np.array([(nx - 1) / 2, (ny - 1) / 2])
    c_out = (d - 1) / 2
    oi, oj = np.meshgrid(np.arange(d) - c_out, np.arange(d) - c_out, indexing="ij")
    m1, p1 = mesh.ravel(), memb.ravel()
    out, worst, worst_cell = [], -math.inf, None
    for ang in np.radians(np.arange(0.0, 360.0, step_deg)):
        ca, sa = math.cos(ang), math.sin(ang)
        ii = np.rint(ca * oi - sa * oj + c_in[0]).astype(int)
        jj = np.rint(sa * oi + ca * oj + c_in[1]).astype(int)
        ok = (ii >= 0) & (ii < nx) & (jj >= 0) & (jj < ny)
        idx = np.where(ok, ii * ny + jj, 0)
        mr, pr = m1[idx] & ok, p1[idx] & ok
        both = mr.any(1) & pr.any(1)
        first_m, first_p = mr[both].argmax(1), pr[both].argmax(1)
        e = (first_m - first_p) * RES
        out.append(e)
        if len(e) and e.max() > worst:
            k = int(np.argmax(e))
            row = np.where(both)[0][k]
            worst, worst_cell = float(e[k]), (int(ii[row, first_p[k]]), int(jj[row, first_p[k]]))
    return (np.concatenate(out) if out else np.zeros(0)), worst_cell


def check_item(model, data, it, zmax, rtop, sight=True):
    shapes = member_shapes(model, data, it["members"])
    verts, faces, bases = [], [], []
    off = 0
    for m in it["meshes"]:
        v, f = fg.model_mesh(m["model"])
        pv = fg.place(v, m["pos"], m["yaw"])
        verts.append(pv)
        faces.append(f + off)
        off += len(v)
        bases.append((m["model"], float(m["pos"][2]), float(pv[:, 2].min())))
    v, f = np.vstack(verts), np.vstack(faces)
    reach = np.array([[c[0], c[1], (max(s[:2]) * math.sqrt(2.0)) if k == "box" else s[0]] for _, k, c, _, s in shapes])
    lo = np.minimum(v[:, :2].min(0), (reach[:, :2] - reach[:, 2:]).min(0)) - 0.05
    hi = np.maximum(v[:, :2].max(0), (reach[:, :2] + reach[:, 2:]).max(0)) + 0.05
    shape = tuple((np.ceil((hi - lo) / RES)).astype(int))
    heights = set(np.round(np.arange(RES / 2, zmax, RES), 4))
    for _, kind, c, _, s in shapes:
        hz = _half_height(kind, s)
        for zz in (c[2] - hz - 0.001, c[2] - hz + 0.001, c[2] + hz - 0.001, c[2] + hz + 0.001):
            if 0 < zz < zmax:
                heights.add(round(float(zz), 4))
    out_cells, worst_out, worst_out_at = 0, 0.0, None
    per_member = {n: {"worst_m": 0.0, "over_5mm": 0, "over_10mm": 0} for n, *_ in shapes}
    worst_surface, worst_surface_at = 0.0, None
    excess_worst, excess_over, excess_lines, excess_at = -math.inf, 0, 0, None
    col_mesh = np.zeros(shape, bool)
    col_memb = np.zeros(shape, bool)
    layers: dict = {}
    for z in sorted(heights):
        mesh = fg.slice_raster(v, f, z, lo, lo + (np.array(shape) - 1) * RES, RES)[:shape[0], :shape[1]]
        memb = member_raster(shapes, z, lo, shape)
        grown = member_raster(shapes, z, lo, shape, PROTRUSION_M - RASTER_M)
        outside = mesh & ~grown
        if outside.any():
            out_cells += int(outside.sum())
            dist = ndimage.distance_transform_edt(~memb) * RES
            o = float(dist[outside].max())
            if o > worst_out:
                c = np.argwhere(outside & (dist >= o - 1e-9))[0]
                worst_out, worst_out_at = o, [round(float(lo[0] + (c[0] + 0.5) * RES), 3),
                                              round(float(lo[1] + (c[1] + 0.5) * RES), 3), z]
        if mesh.any() and memb.any():
            key = (np.packbits(mesh).tobytes(), np.packbits(memb).tobytes())
            if key in layers:
                layers[key][2] += 1
            else:
                layers[key] = [mesh, memb, 1, z]
        if z <= rtop:
            col_mesh |= mesh
            col_memb |= memb
    names = [n for n, *_ in shapes]
    for mesh, memb, count, z in layers.values():  # identical heights give identical results
        cells, dist = surface_cells(mesh, memb)
        if len(cells):
            owner = member_raster(shapes, z, lo, mesh.shape, label=True)[cells[:, 0], cells[:, 1]]
            for k, n in enumerate(names):
                d = dist[owner == k + 1]
                if len(d):
                    pm = per_member[n]
                    pm["worst_m"] = round(max(pm["worst_m"], float(d.max()) + RASTER_M), 4)
                    pm["over_5mm"] += count * int(np.sum(d + RASTER_M > SURFACE_TARGET_M))
                    pm["over_10mm"] += count * int(np.sum(d + RASTER_M > 0.01))
            k = int(np.argmax(dist))
            if dist[k] > worst_surface:
                worst_surface = float(dist[k])
                worst_surface_at = [round(float(lo[0] + (cells[k][0] + 0.5) * RES), 3),
                                    round(float(lo[1] + (cells[k][1] + 0.5) * RES), 3), z]
        e, cell = sight_line_excess(mesh, memb) if sight else (np.zeros(0), None)
        if len(e):
            excess_lines += count * len(e)
            if float(e.max()) > excess_worst:
                excess_worst = float(e.max())
                owner = member_raster(shapes, z, lo, mesh.shape, label=True)[cell]
                excess_at = {"height_m": z, "member": names[owner - 1] if owner else None}
            excess_over += count * int(np.sum(e > SURFACE_M))
    hidden_open = col_memb & ~col_mesh  # solid members where the mesh is open to the robot's top
    room = float((ndimage.distance_transform_edt(hidden_open) * RES).max()) + RASTER_M if hidden_open.any() else 0.0
    member_tops = [c[2] + _half_height(k, s) for _, k, c, _, s in shapes]
    support = []
    for mid, base_z, low_z in bases:  # its lowest point is on the floor or on one of its members' tops
        stands_on = min([0.0] + member_tops, key=lambda t: abs(t - low_z))
        support.append({"mesh": mid, "error_m": round(abs(low_z - stands_on), 4)})
    res = {"item": it["name"], "members": len(shapes), "faces": int(len(f)),
           "cells_outside_members_1cm": out_cells, "max_protrusion_m": round(worst_out, 4),
           "protrusion_at": worst_out_at,
           "max_member_to_mesh_m": round(worst_surface + RASTER_M, 4), "member_to_mesh_at": worst_surface_at,
           "members_over_5mm": {n: pm for n, pm in per_member.items() if pm["over_5mm"]},
           "sight_lines_meeting_both": excess_lines,
           "sight_lines_mesh_over_2cm_behind": excess_over,
           "sight_lines_share_over_2cm": round(excess_over / excess_lines, 4) if excess_lines else None,
           "sight_line_worst_excess_m": round(excess_worst, 4) if excess_worst > -math.inf else None,
           "sight_line_worst_at": excess_at,
           "distinct_layers": len(layers), "hidden_open_room_m": round(room, 4), "support": support}
    res["pass"] = (out_cells == 0 and worst_surface + RASTER_M <= SURFACE_M and room < FOOTPRINT_HALF_MIN
                   and all(s["error_m"] <= SUPPORT_M for s in support))
    return res


def lidar_gate(model, data, rtop, plane) -> list[dict]:
    """Every solid static geom reaching below the robot's top: spans the lidar plane with margin,
    or lies (seen from above) inside such solids grown by COVER_GROW_M. Returns the failures."""
    import mujoco
    robot = model.body("robot").id
    floor = model.geom("floor").id
    seen, low = [], []
    for g in range(model.ngeom):
        if g == floor or model.geom_contype[g] == 0 or model.body_rootid[model.geom_bodyid[g]] == robot:
            continue
        b = model.geom_bodyid[g]
        if model.body_weldid[b] != 0 or model.body_mocapid[b] >= 0:
            continue  # moving actors (cats) and markers are not furniture
        z0, z1 = geom_z_extent(model, data, g)
        if z0 >= rtop:
            continue
        if z0 <= plane - PLANE_MARGIN_M and z1 >= plane + PLANE_MARGIN_M:
            seen.append(g)
        else:
            low.append(g)

    def shape_of(g):
        if model.geom_type[g] not in (mujoco.mjtGeom.mjGEOM_BOX, mujoco.mjtGeom.mjGEOM_CYLINDER):
            raise ValueError(f"lidar gate: {model.geom(g).name} is neither a box nor a cylinder")
        kind = "box" if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX else "cyl"
        xm = data.geom_xmat[g].reshape(3, 3)
        if abs(xm[2, 2] - 1.0) > 1e-9:
            raise ValueError(f"lidar gate: {model.geom(g).name} is not upright (its plan is not drawn)")
        return ("", kind, data.geom_xpos[g].copy(), math.atan2(xm[1, 0], xm[0, 0]), model.geom_size[g].copy())

    def bounds(g, grow):
        size = model.geom_size[g]
        rr = (math.hypot(size[0], size[1]) if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX else size[0]) + grow
        x, y = data.geom_xpos[g][:2]
        return np.array([x - rr, y - rr]), np.array([x + rr, y + rr])

    def plan(sh, lo, shape, grow):
        flat = (sh[0], sh[1], np.array([sh[2][0], sh[2][1], 0.0]), sh[3], sh[4])
        return member_raster([flat], 0.0, lo, shape, grow)

    fails = []
    for g in low:
        lo, hi = bounds(g, 0.01)
        shape = tuple((np.ceil((hi - lo) / RES)).astype(int))
        need = plan(shape_of(g), lo, shape, 0.0)
        cover = np.zeros(shape, bool)
        for s in seen:
            slo, shi = bounds(s, COVER_GROW_M)
            if np.any(shi < lo) or np.any(slo > hi):
                continue
            cover |= plan(shape_of(s), lo, shape, COVER_GROW_M - RASTER_M)
        miss = need & ~cover
        if miss.any():
            z0, z1 = geom_z_extent(model, data, g)
            fails.append({"geom": model.geom(g).name, "z": [round(z0, 4), round(z1, 4)],
                          "uncovered_cells": int(miss.sum())})
    return fails


def overlap_gate(model, data, items, step=0.005, allow_m=0.002, extra_models=()) -> list[dict]:
    """Nothing drawn passes through anything else: at every height (5 mm steps, plus 1 mm inside
    each mesh's bottom and top and on both sides of every static solid's faces, so a thin plate
    cannot fall between samples) up to each mesh's top, every placed mesh's filled section must not
    overlap the filled section of any other placed mesh (of any item, its own included: a planter
    on its stand; models placed outside items too: extra_models) or any static solid other than its
    own item's members (walls, doors, other items' members), beyond contact (allow_m plus the
    raster: each section's outline cells are contact, not overlap). A mesh is not compared with
    itself."""
    import mujoco
    robot = model.body("robot").id
    floor = model.geom("floor").id
    placed = []  # (item name or None, mesh model id, vertices, faces)
    for it in items:
        for m in it["meshes"]:
            v, f = fg.model_mesh(m["model"])
            placed.append((it["name"], m["model"], fg.place(v, m["pos"], m["yaw"]), f))
    for m in extra_models:
        v, f = fg.model_mesh(m["model"])
        placed.append((None, m["model"], fg.place(v, m["pos"], m["yaw"]), f))
    member_owner = {n: it["name"] for it in items for n in it["members"]}
    statics = []
    for g in range(model.ngeom):
        b = model.geom_bodyid[g]
        if g == floor or model.geom_contype[g] == 0 or model.body_rootid[b] == robot:
            continue
        if model.body_weldid[b] != 0 or model.body_mocapid[b] >= 0:
            continue
        if model.geom_type[g] not in (mujoco.mjtGeom.mjGEOM_BOX, mujoco.mjtGeom.mjGEOM_CYLINDER):
            raise ValueError(f"overlap gate: {model.geom(g).name} is neither a box nor a cylinder")
        xm = data.geom_xmat[g].reshape(3, 3)
        if abs(xm[2, 2] - 1.0) > 1e-9:
            raise ValueError(f"overlap gate: {model.geom(g).name} is not upright")
        kind = "box" if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX else "cyl"
        statics.append((model.geom(g).name, kind, data.geom_xpos[g].copy(), math.atan2(xm[1, 0], xm[0, 0]),
                        model.geom_size[g].copy()))
    fails = []
    for k, (owner, mid, v, f) in enumerate(placed):
        lo, hi = v[:, :2].min(0) - 0.02, v[:, :2].max(0) + 0.02
        shape = tuple((np.ceil((hi - lo) / RES)).astype(int))
        top = lo + (np.array(shape) - 1) * RES
        foreign = [o for o in statics if member_owner.get(o[0]) != owner]
        found = None
        z_lo, z_hi = float(v[:, 2].min()), float(v[:, 2].max())
        heights = set(np.round(np.arange(z_lo + step / 2, z_hi, step), 4)) | {round(z_lo + 0.001, 4), round(z_hi - 0.001, 4)}
        for _, kind, c, _, sz in foreign:
            hz = _half_height(kind, sz)
            for zz in (c[2] - hz - 0.001, c[2] - hz + 0.001, c[2] + hz - 0.001, c[2] + hz + 0.001):
                if z_lo < zz < z_hi:
                    heights.add(round(float(zz), 4))
        for z in sorted(heights):
            mine = fg.slice_raster(v, f, z, lo, top, RES)[:shape[0], :shape[1]]
            if not mine.any():
                continue
            core = ndimage.binary_erosion(mine, iterations=1)  # its outline cells are contact
            hit = member_raster(foreign, z, lo, shape, -(allow_m + RASTER_M)) & core
            for j, (owner2, mid2, v2, f2) in enumerate(placed):
                if j == k or v2[:, 2].min() > z or v2[:, 2].max() < z:
                    continue
                if v2[:, 0].max() < lo[0] or v2[:, 0].min() > hi[0] or v2[:, 1].max() < lo[1] or v2[:, 1].min() > hi[1]:
                    continue
                theirs = fg.slice_raster(v2, f2, z, lo, top, RES)[:shape[0], :shape[1]]
                hit |= ndimage.binary_erosion(theirs, iterations=1) & core
            if hit.any():
                c = np.argwhere(hit)[0]
                found = {"item": owner, "mesh": mid, "z": round(float(z), 3), "cells": int(hit.sum()),
                         "at": [round(float(lo[0] + (c[0] + 0.5) * RES), 3), round(float(lo[1] + (c[1] + 0.5) * RES), 3)]}
                break
        if found:
            fails.append(found)
    return fails


def main() -> int:
    sim = compiled_world()
    model, data = sim.model, sim.data
    rtop, plane = robot_top_and_plane(model, data)
    zmax = max(0.40, rtop + 0.05)
    sight = "--no-sight" not in sys.argv  # the sight-line diagnostic is slow; skip it while fitting
    wanted = [a for a in sys.argv[1:] if not a.startswith("--")]
    results = []
    for it in build_world.furniture_items:
        if wanted and it["name"] not in wanted:
            continue
        r = check_item(model, data, it, zmax, rtop, sight)
        results.append(r)
        print(("PASS " if r["pass"] else "FAIL ") + json.dumps(r))
    lidar = lidar_gate(model, data, rtop, plane)
    print(("PASS" if not lidar else "FAIL") + f" lidar gate (robot top {rtop:.4f} m, plane {plane:.4f} m): {lidar}")
    overlap = overlap_gate(model, data, build_world.furniture_items,
                           extra_models=[m for m in build_world.placed_models if m["item"] is None])
    print(("PASS" if not overlap else "FAIL") + f" overlap gate (nothing drawn passes into anything else): {overlap}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"gate_height_m": zmax, "robot_top_m": rtop, "lidar_plane_m": plane,
                               "items": results, "lidar_gate_failures": lidar, "overlap_failures": overlap},
                              indent=1), encoding="utf-8")
    sim.close()
    return 0 if all(r["pass"] for r in results) and not lidar and not overlap else 1


if __name__ == "__main__":
    raise SystemExit(main())
