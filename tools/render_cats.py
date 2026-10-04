"""Comparison renders of the cat coats (and poses) on a neutral floor. Development check.

    .venv\\Scripts\\python tools\\render_cats.py            # writes qa_output/cats/coats.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from robot_env.cat_rig import Skeleton, model_parts  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "qa_output" / "cats"
COATS = {"brown_tabby": "cat_anisotropic1.png", "ginger_mackerel": "coat_ginger_mackerel.png",
         "black_tuxedo": "coat_black_tuxedo.png", "grey_tabby": "coat_grey_tabby.png"}
VIEWS = (("front", 180, -8), ("side", 90, -8), ("rear", 0, -10), ("three_quarter", 135, -18))


def build(coat_file: str):
    tex = {"anisotropic1": coat_file, "anisotropic2": "cat_anisotropic2.png", "anisotropic3": "cat_anisotropic3.png"}
    a, b, s, files = model_parts("c_", tex)
    xml = f"""<mujoco><asset>{a}<texture name="grid" type="2d" builtin="checker" rgb1=".62 .6 .57" rgb2=".58 .56 .53"
      width="64" height="64"/><material name="floor" texture="grid" texrepeat="10 10"/></asset>
      <visual><headlight ambient=".35 .35 .35" diffuse=".45 .45 .45"/><global offwidth="1280" offheight="720"/>
      <quality shadowsize="4096"/></visual>
      <worldbody><light pos="1.2 -1 2.2" dir="-1.2 1 -2.2" diffuse=".65 .65 .62" castshadow="true"/>
      <light pos="-1 1.5 1.8" dir="1 -1.5 -1.8" diffuse=".25 .25 .28" castshadow="false"/>
      <geom type="plane" size="3 3 .1" material="floor"/>{b}</worldbody><deformable>{s}</deformable></mujoco>"""
    m = mujoco.MjModel.from_xml_string(xml, files)
    return m, mujoco.MjData(m)


def render_sheet(local_pose=None, name="coats.png") -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    W, H = 480, 300
    sheet = Image.new("RGB", (W * len(VIEWS), H * len(COATS)), (20, 20, 20))
    for row, (coat, f) in enumerate(COATS.items()):
        m, d = build(f)
        sk = Skeleton(m, "c_")
        sk.pose(np.zeros(3), np.eye(3), local_pose(sk.rig) if local_pose else None)
        sk.write(d)
        mujoco.mj_forward(m, d)
        r = mujoco.Renderer(m, H, W)
        for col, (vname, az, el) in enumerate(VIEWS):
            cam = mujoco.MjvCamera()
            cam.lookat[:] = (-0.04, 0, 0.15)
            cam.distance, cam.azimuth, cam.elevation = 0.95, az, el
            r.update_scene(d, cam)
            im = Image.fromarray(r.render())
            ImageDraw.Draw(im).text((6, 6), f"{coat} / {vname}", fill=(255, 255, 0))
            sheet.paste(im, (col * W, row * H))
        r.close()
    path = OUT / name
    sheet.save(path)
    return path


if __name__ == "__main__":
    print(render_sheet())
