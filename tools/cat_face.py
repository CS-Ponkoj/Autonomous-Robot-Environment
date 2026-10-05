"""The cats' faces and eyes: close-ups of all four coats and measured eye symmetry (development
evidence for the eye gate).

    .venv\\Scripts\\python tools\\cat_face.py         -> qa_output/cats/faces.png, faces_robot_camera.png

faces.png: one row per coat; for each of three lights (overhead, from the left, from the right),
three columns: 45 degrees right, front, 45 degrees left (white gutters between them). faces_robot_camera.png: the robot's own camera
looking at each cat from 0.5, 1.0, and 1.5 m. Printed per coat, from the skinned mesh as drawn:
the height difference of the two irises (pupils level), their size difference, and the angle
between the two gaze directions.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from robot_env.cats import COAT_NAMES  # noqa: E402
from robot_env.system import RobotSystem  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "qa_output" / "cats" / "faces.png"
ROBOT_OUT = OUT.with_name("faces_robot_camera.png")
SIZE = 300
IRIS_PRIMS = (3, 4)  # left and right iris skins of each cat
LIGHTS = {"overhead": (0.0, 0.0), "from the left": (-1.2, 0.0), "from the right": (1.2, 0.0)}


def _herd() -> RobotSystem:
    s = RobotSystem(cats=4, cat_seed=16)
    s.reset(-4.3, 0.0, 0.0, (4.5, 0.0))
    for i in range(4):  # in a row along the corridor, facing +y
        s.cats.place(i, -2.4 + 1.6 * i, -0.2, math.pi / 2, state="pause")
    s.sim.before_render()
    return s


def _eye_centre(s, i):
    names = s.cats.rig.joint_names
    d, m = s.sim.data, s.sim.model
    eyes = [d.xpos[m.body(f"cat{i}_b{names.index(n)}").id] for n in ("Character1_EyeL_00", "Character1_EyeR_025")]
    return (eyes[0] + eyes[1]) / 2


def eye_metrics(s, r) -> list[dict]:
    """From the skin as drawn: iris centres, sizes, and gaze directions (centre of the iris mesh
    relative to its eyeball's bone)."""
    m, d = s.sim.model, s.sim.data
    sc = r.scene
    out = []
    names = s.cats.rig.joint_names
    for i in range(4):
        cen, size, gaze = [], [], []
        for prim, bone in zip(IRIS_PRIMS, ("Character1_EyeL_00", "Character1_EyeR_025")):
            k = m.skin(f"cat{i}_skin{prim}").id
            a, n = sc.skinvertadr[k], sc.skinvertnum[k]
            v = sc.skinvert[3 * a: 3 * (a + n)].reshape(-1, 3)
            c = v.mean(0)
            cen.append(c)
            size.append(float(np.linalg.norm(v - c, axis=1).max()))
            g = c - d.xpos[m.body(f"cat{i}_b{names.index(bone)}").id]
            gaze.append(g / np.linalg.norm(g))
        out.append({
            "coat": COAT_NAMES[i],
            "level_mm": round(1000 * abs(cen[0][2] - cen[1][2]), 2),
            "size_diff_pct": round(100 * abs(size[0] - size[1]) / max(size), 2),
            "gaze_angle_deg": round(float(np.degrees(np.arccos(np.clip(gaze[0] @ gaze[1], -1, 1)))), 1),
        })
    return out


def _grid(rows: list[list[np.ndarray]], gap: int = 10, group: int = 3, group_gap: int = 40) -> np.ndarray:
    """Tiles laid out with white gutters between them (wider between groups of `group` tiles: one
    lighting), so neighbouring views never read as one picture."""
    h, w = rows[0][0].shape[:2]
    xs, x = [], 0
    for j in range(len(rows[0])):
        xs.append(x)
        x += w + (group_gap if (j + 1) % group == 0 else gap)
    width = xs[-1] + w
    out = np.full((len(rows) * (h + gap) - gap, width, 3), 255, dtype=np.uint8)
    for i, row in enumerate(rows):
        for j, tile in enumerate(row):
            out[i * (h + gap):i * (h + gap) + h, xs[j]:xs[j] + w] = tile
    return out


def main() -> int:
    s = _herd()
    m, d = s.sim.model, s.sim.data
    light = m.light("light_corridor_w").id
    base_pos, base_dir = m.light_pos[light].copy(), m.light_dir[light].copy()
    rows = []
    with mujoco.Renderer(m, SIZE, SIZE) as r:
        r.update_scene(d, mujoco.MjvCamera())
        metrics = eye_metrics(s, r)
        for i in range(4):
            head = _eye_centre(s, i)
            tiles = []
            for side, _ in LIGHTS.values():
                m.light_pos[light] = head + np.array([side, -0.8, 0.9])  # a lamp in front, high, left or right
                dvec = head - m.light_pos[light]
                m.light_dir[light] = dvec / np.linalg.norm(dvec)
                for azim in (-135.0, -90.0, -45.0):  # 45 right, front, 45 left (cats face +y): heads toward the middle
                    cam = mujoco.MjvCamera()
                    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
                    cam.lookat[:] = head
                    cam.distance, cam.azimuth, cam.elevation = 0.24, azim, -6
                    r.update_scene(d, cam)
                    tiles.append(r.render())
            rows.append(tiles)
        m.light_pos[light], m.light_dir[light] = base_pos, base_dir
    OUT.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(_grid(rows)).save(OUT)
    # the robot's own camera at 0.5, 1.0, and 1.5 m in front of each cat
    cam_rows = []
    for i in range(4):  # each cat in turn mid-corridor, facing the robot down the corridor
        for j in range(4):
            s.cats.place(j, (0.5, -4.0, 4.0, -2.5)[j] if j != i else 0.5, 0.0 if j == i else 0.45,
                         math.pi, state="pause")
        tiles = []
        for dist in (0.5, 1.0, 1.5):
            s.sim.reset(0.5 - 0.25 - dist, 0.0, 0.0, (4.5, 0.0))  # dist from the cat's nose to the robot
            s.cats.write_bones()
            tiles.append(s.sim.render_camera((240, 320)))
        cam_rows.append(tiles)
    Image.fromarray(_grid(cam_rows)).save(ROBOT_OUT)
    for row in metrics:
        print(row)
    print("wrote", OUT, "and", ROBOT_OUT)
    s.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
