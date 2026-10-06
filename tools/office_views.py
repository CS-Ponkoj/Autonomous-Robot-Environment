"""Eye-height views of the office, or of the reception (ceiling on), for realism reviews:
qa_output/realism/office/ or qa_output/realism/reception/.

    .venv\Scripts\python tools\office_views.py [reception]
"""

from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from robot_env.sim import RobotSim  # noqa: E402

OUT = ROOT / "qa_output" / "realism" / "office"
W, H = 1280, 720
VIEWS = {  # name: (look-at, azimuth, elevation, distance)
    "desk_from_door": ((-3.8, 4.2, 0.7), 131.5, -14.0, 3.5),
    "shelf_and_lamp": ((-4.6, 2.8, 0.9), 160.0, -10.0, 2.6),
    "cabinet_and_plant": ((-0.6, 2.95, 0.7), 0.0, -10.0, 2.7),
    "desk_close": ((-3.8, 4.3, 0.75), 95.0, -22.0, 1.7),
    "chair_close": ((-3.8, 3.62, 0.55), 70.0, -18.0, 1.4),  # close-ups, one item each
    "books_close": ((-4.62, 2.4, 1.15), 180.0, -5.0, 1.3),
    "bin_close": ((-2.6, 4.75, 0.25), 120.0, -22.0, 1.0),
    "lamp_base_close": ((-4.78, 3.85, 0.3), 120.0, -20.0, 1.0),
    "stand_close": ((-0.45, 1.25, 0.35), 0.0, -20.0, 1.3),
    "cabinet_close": ((-0.45, 4.3, 0.9), 75.0, -15.0, 1.5),
}


RECEPTION = {  # (office_views.py reception): the reception's seating, from the door and close
    "reception_from_door": ((2.0, -3.7, 0.5), -95.0, -16.0, 3.2),
    "reception_seating": ((1.6, -4.0, 0.45), -60.0, -14.0, 2.6),
    "reception_armchair": ((2.5, -3.6, 0.45), 200.0, -15.0, 1.5),
    "reception_corner": ((4.1, -4.5, 0.4), -110.0, -15.0, 1.8),
}


def main() -> int:
    global OUT, VIEWS
    if sys.argv[1:] == ["reception"]:
        OUT, VIEWS = OUT.parent / "reception", RECEPTION
    OUT.mkdir(parents=True, exist_ok=True)
    sim = RobotSim()
    sim.reset(-1.2, 2.0, 2.4, (2.0, 2.0))
    r = mujoco.Renderer(sim.model, H, W)
    opt = mujoco.MjvOption()
    opt.geomgroup[3] = 1  # the ceiling, as a person in the room sees it
    tiles = []
    for name, (look, az, el, dist) in VIEWS.items():
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = look
        cam.azimuth, cam.elevation, cam.distance = az, el, dist
        r.update_scene(sim.data, cam, scene_option=opt)
        img = r.render()
        Image.fromarray(img).save(OUT / f"{name}.png")
        tiles.append(img)
    sheet = np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:4])])
    Image.fromarray(sheet).save(OUT / "_sheet.png")
    if len(tiles) > 4:
        close = [Image.fromarray(t).resize((W // 2, H // 2), Image.LANCZOS) for t in tiles[4:]]
        sheet = np.vstack([np.hstack([np.asarray(c) for c in close[k:k + 3]]) for k in (0, 3)])
        Image.fromarray(sheet).save(OUT / "_close_sheet.png")
    r.close()
    sim.close()
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
