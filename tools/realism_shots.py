"""Fixed-pose renders of every room for judging how real the environment looks (development
tool): each room from a person's eye height (ceiling shown, as a person sees it), the corridor
from the robot's height, and close-ups of wall, floor, door, and window details. The same poses
every time, so before and after can be compared shot for shot.

    .venv\\Scripts\\python tools\\realism_shots.py [label]   (writes qa_output/realism/<label>/)
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from robot_env.system import RobotSystem  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
W, H = 1280, 720
FOVY = 60.0
# name: (eye x, y, z), (look at x, y, z)
SHOTS = {
    "office_from_door": ((-0.6, 1.15, 1.55), (-3.6, 3.9, 0.6)),
    "office_desk": ((-4.6, 1.2, 1.45), (-2.2, 4.6, 0.8)),
    "lab_from_door": ((0.6, 1.15, 1.55), (3.3, 3.9, 0.7)),
    "lab_benches": ((4.6, 1.2, 1.45), (1.4, 4.3, 0.8)),
    "storage_aisles": ((-0.5, -1.15, 1.55), (-3.6, -3.9, 0.7)),
    "reception_from_door": ((4.6, -1.15, 1.55), (1.4, -4.1, 0.6)),
    "reception_sofa": ((0.6, -1.15, 1.45), (3.4, -3.8, 0.6)),
    "corridor": ((-4.6, 0.0, 1.55), (4.0, 0.0, 0.9)),
    "corridor_robot_height": ((-4.2, 0.0, 0.25), (3.0, 0.0, 0.45)),
    "detail_wall_corner": ((-1.2, 1.45, 0.75), (-0.15, 0.95, 0.25)),
    "detail_door": ((-2.5, -0.35, 1.25), (-2.5, 0.95, 1.0)),
    "detail_window": ((-3.2, 3.6, 1.4), (-3.8, 5.0, 1.5)),
    "detail_floor_rug": ((-2.6, 2.6, 0.9), (-3.8, 3.9, 0.0)),
}


def render(label: str) -> Path:
    out = ROOT / "qa_output" / "realism" / label
    out.mkdir(parents=True, exist_ok=True)
    s = RobotSystem(cats=0)
    s.reset(4.4, -4.4, math.pi / 4, (4.0, 4.0))  # the robot parked in the reception corner
    m, d = s.sim.model, s.sim.data
    m.vis.global_.fovy = FOVY
    r = mujoco.Renderer(m, height=H, width=W)
    opt = mujoco.MjvOption()
    opt.geomgroup[3] = 1  # the ceiling and its lights, as a person standing in the room sees them
    light_pos = np.array(m.light_pos)
    tiles = []
    for name, (eye, at) in SHOTS.items():
        eye, at = np.array(eye, dtype=float), np.array(at, dtype=float)
        v = at - eye
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = at
        cam.distance = float(np.linalg.norm(v))
        cam.azimuth = math.degrees(math.atan2(v[1], v[0]))
        cam.elevation = math.degrees(math.asin(v[2] / np.linalg.norm(v)))
        # shadows from the two room lights nearest the eye (as the window does)
        m.light_castshadow[:] = 0
        m.light_castshadow[np.argsort(np.linalg.norm(light_pos - eye, axis=1))[:2]] = 1
        s.sim.before_render()
        r.update_scene(d, cam, opt)
        r.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 1
        r.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 1
        img = r.render().copy()
        from PIL import Image
        Image.fromarray(img).save(out / f"{name}.png", optimize=True)
        tiles.append(Image.fromarray(img).resize((W // 4, H // 4)))
        print(name)
    from PIL import Image
    cols = 4
    sheet = Image.new("RGB", (cols * (W // 4), math.ceil(len(tiles) / cols) * (H // 4)), "white")
    for k, t in enumerate(tiles):
        sheet.paste(t, ((k % cols) * (W // 4), (k // cols) * (H // 4)))
    sheet.save(out / "_sheet.png", optimize=True)
    r.close()
    s.close()
    return out


if __name__ == "__main__":
    print(render(sys.argv[1] if len(sys.argv) > 1 else "latest"))
