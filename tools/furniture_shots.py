"""Close approach views of every furniture item (development check), tiled into
qa_output/furniture/approach.png:

    .venv\\Scripts\\python tools\\furniture_shots.py [item names]

For each item the robot is parked where it fits (inflated planning grid), STANDOFF metres from the
item's nearest member, facing the item's centre, with a clear line from its camera to the item.
Two images: the robot camera, and a chase view from behind and above the robot, pulled in until
nothing blocks its view of the robot or the item. Segmentation renders confirm that the item (and,
in the chase view, the robot) is really visible; the run fails if not.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import build_world  # noqa: E402
from robot_env.layout import RoomMap, Shape, clearance  # noqa: E402
from robot_env.sim import RobotSim  # noqa: E402

OUT = ROOT / "qa_output" / "furniture" / "approach.png"
W, H = 480, 360
STANDOFF = (0.55, 0.7, 0.85, 1.0, 1.2)  # tried in order
MIN_SHARE = 0.01  # the item fills at least 1% of each image (the robot too, in the chase view)
_VIEW = np.array([1, 0, 1, 0, 0, 1], dtype=np.uint8)


def item_geoms(model, it):
    """Geom ids of the item's members and of its mesh geoms (meshes from its models)."""
    ids = {model.geom(n).id for n in it["members"]}
    prefixes = tuple(f'{it["name"]}_{m["model"]}_' for m in it["meshes"])  # model() names them so
    return ids | {g for g in range(model.ngeom) if model.geom(g).name.startswith(prefixes)}


def member_shapes(sim, it):
    m, d = sim.model, sim.data
    out = []
    for n in it["members"]:
        g = m.geom(n).id
        x, y = d.geom_xpos[g][:2]
        xm = d.geom_xmat[g].reshape(3, 3)
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX:
            out.append(Shape("box", float(x), float(y), math.atan2(xm[1, 0], xm[0, 0]), float(m.geom_size[g][0]),
                             float(m.geom_size[g][1])))
        else:
            out.append(Shape("cylinder", float(x), float(y), 0.0, float(m.geom_size[g][0]), float(m.geom_size[g][0])))
    return out


def first_hit(sim, origin, target):
    d = np.asarray(target, float) - np.asarray(origin, float)
    dist = float(np.linalg.norm(d))
    gid = np.zeros(1, dtype=np.int32)
    hit = mujoco.mj_ray(sim.model, sim.data, np.asarray(origin, float), d / dist, _VIEW, 1, -1, gid)
    return (int(gid[0]) if hit >= 0 else -1), hit, dist


def approach_pose(sim, room, it, ids):
    shapes = member_shapes(sim, it)
    centre = np.mean([[s.x, s.y] for s in shapes], axis=0)
    for standoff in STANDOFF:
        for a in np.radians(np.arange(0, 360, 5)):
            for r in np.arange(0.2, 2.5, 0.02):
                x, y = centre[0] + r * math.cos(a), centre[1] + r * math.sin(a)
                if float(clearance(shapes, np.array(x), np.array(y))) >= standoff:
                    break
            else:
                continue
            if not (abs(x) < 4.8 and abs(y) < 4.8 and room.free[room.cell(x, y)]):
                continue
            g, _, _ = first_hit(sim, (x, y, 0.3), (centre[0], centre[1], 0.3))
            if g in ids:
                return x, y, math.atan2(centre[1] - y, centre[0] - x), centre, standoff
    return None


def chase_camera(sim, robot_xy, yaw, centre, ids):
    """Behind and above the robot, as far back as the view of the robot and the item stays clear."""
    look = np.array([0.5 * (robot_xy[0] + centre[0]), 0.5 * (robot_xy[1] + centre[1]), 0.25])
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = look
    cam.azimuth = math.degrees(yaw)
    cam.elevation = -25
    robot = sim.model.body("robot").id
    for dist in np.arange(1.8, 0.5, -0.05):
        el = math.radians(cam.elevation)
        fwd = np.array([math.cos(el) * math.cos(math.radians(cam.azimuth)),
                        math.cos(el) * math.sin(math.radians(cam.azimuth)), math.sin(el)])
        eye = look - dist * fwd
        g_robot, _, _ = first_hit(sim, eye, (robot_xy[0], robot_xy[1], 0.1))
        g_item, _, _ = first_hit(sim, eye, (centre[0], centre[1], 0.3))
        robot_seen = g_robot >= 0 and sim.model.body_rootid[sim.model.geom_bodyid[g_robot]] == robot
        if robot_seen and g_item in ids:
            cam.distance = dist
            return cam
    cam.distance = 0.6
    return cam


def share(seg, ids):
    return float(np.isin(seg[..., 0], list(ids)).mean())


def main() -> int:
    build_world.generate()
    wanted = sys.argv[1:]
    items = [it for it in build_world.furniture_items if not wanted or it["name"] in wanted]
    sim = RobotSim()
    room = RoomMap(sim.model)
    view = mujoco.Renderer(sim.model, H, W)
    seg = mujoco.Renderer(sim.model, H, W)
    seg.enable_segmentation_rendering()
    robot = sim.model.body("robot").id
    robot_ids = {g for g in range(sim.model.ngeom) if sim.model.body_rootid[sim.model.geom_bodyid[g]] == robot}
    tiles, problems = [], []
    for it in items:
        ids = item_geoms(sim.model, it)
        pose = approach_pose(sim, room, it, ids)
        if pose is None:
            problems.append(f"{it['name']}: no clear approach pose")
            continue
        x, y, yaw, centre, standoff = pose
        sim.reset(x, y, yaw, (0.0, 0.0))
        mujoco.mj_forward(sim.model, sim.data)
        cam_img = sim.render_camera((H, W))
        cam = chase_camera(sim, (x, y), yaw, centre, ids)
        view.update_scene(sim.data, cam)
        chase_img = view.render()
        seg.update_scene(sim.data, cam)
        s = seg.render()
        item_share, robot_share = share(s, ids), share(s, robot_ids)
        if item_share < MIN_SHARE or robot_share < MIN_SHARE:
            problems.append(f"{it['name']}: chase view shows item {item_share:.1%}, robot {robot_share:.1%}")
        tile = Image.new("RGB", (2 * W, H + 24), "white")
        tile.paste(Image.fromarray(cam_img), (0, 24))
        tile.paste(Image.fromarray(chase_img), (W, 24))
        ImageDraw.Draw(tile).text((6, 5), f"{it['name']}: robot {standoff:.2f} m from it; robot camera (left), "
                                          f"chase (right; item {item_share:.0%}, robot {robot_share:.0%})", fill="black")
        tiles.append(tile)
    sheet = Image.new("RGB", (2 * W, max(1, len(tiles)) * (H + 24)), "white")
    for k, t in enumerate(tiles):
        sheet.paste(t, (0, k * (H + 24)))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(OUT)
    view.close()
    seg.close()
    sim.close()
    print("wrote", OUT, len(tiles), "items")
    for p in problems:
        print("PROBLEM", p)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
