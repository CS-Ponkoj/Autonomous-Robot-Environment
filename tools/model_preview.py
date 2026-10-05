"""Labelled previews of the converted furniture models (development check).

    .venv\\Scripts\\python tools\\model_preview.py [model ids]       -> qa_output/furniture/preview.png
    .venv\\Scripts\\python tools\\model_preview.py --parity [ids]    (.msh against the same data as OBJ)

Each model is drawn on a neutral floor from the front three-quarter and the back three-quarter,
labelled with its id, faces, and size. --parity renders every model twice, once from its .msh and
once from an OBJ written from the same arrays (OBJ texture rows are flipped by MuJoCo itself), and
reports the share of pixels that differ by more than 2 levels (at most 0.01%, shadow-edge rounding of
the OBJ text): the binary meshes must show exactly what the source shows.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from fetch_models import read_msh  # noqa: E402

FURN = ROOT / "robot_env" / "assets" / "furniture"
OUT = ROOT / "qa_output" / "furniture" / "preview.png"
TILE = 360
LABEL = 40


def _obj_bytes(data: bytes) -> bytes:
    v, n, uv, f = read_msh(data)
    uv = np.column_stack([uv[:, 0], 1.0 - uv[:, 1]])  # back to glTF rows: MuJoCo flips OBJ rows
    lines = [f"v {a:.7g} {b:.7g} {c:.7g}" for a, b, c in v]
    lines += [f"vt {a:.7g} {b:.7g}" for a, b in uv]
    lines += [f"vn {a:.7g} {b:.7g} {c:.7g}" for a, b, c in n]
    lines += [f"f {a + 1}/{a + 1}/{a + 1} {b + 1}/{b + 1}/{b + 1} {c + 1}/{c + 1}/{c + 1}" for a, b, c in f]
    return ("\n".join(lines) + "\n").encode()


def scene_xml(model_id: str, as_obj: bool = False) -> tuple[str, dict]:
    rec = json.loads((FURN / f"{model_id}.json").read_text(encoding="utf-8"))
    assets, geoms, files = [], [], {}
    for k, p in enumerate(rec["parts"]):
        data = (FURN / p["mesh"]).read_bytes()
        name = p["mesh"]
        if as_obj:
            name = name.replace(".msh", ".obj")
            data = _obj_bytes(data)
        files[name] = data
        assets.append(f'<mesh name="m{k}" file="{name}" inertia="shell"/>')
        if p["texture"]:
            files[p["texture"]] = (FURN / p["texture"]).read_bytes()
            assets.append(f'<texture name="t{k}" type="2d" file="{p["texture"]}"/>')
            tex = f'texture="t{k}"'
        else:
            tex = f'rgba="{" ".join(str(c) for c in p["rgba"])}"'
        assets.append(f'<material name="mat{k}" {tex} specular="{p["specular"]}" shininess="{p["shininess"]}" '
                      f'reflectance="{p["reflectance"]}"/>')
        geoms.append(f'<geom type="mesh" mesh="m{k}" material="mat{k}" contype="0" conaffinity="0"/>')
    xml = f"""<mujoco><visual><global offwidth="{TILE}" offheight="{TILE}"/><quality shadowsize="2048"/>
<headlight ambient="0.35 0.35 0.35" diffuse="0.25 0.25 0.25"/></visual>
<asset><texture name="grid" type="2d" builtin="checker" rgb1=".75 .75 .73" rgb2=".7 .7 .68" width="64" height="64"/>
<material name="floor" texture="grid" texrepeat="8 8"/>{''.join(assets)}</asset>
<worldbody><light pos="1.5 -2 3" dir="-0.4 0.5 -1" castshadow="true" diffuse="0.6 0.6 0.6"/>
<geom type="plane" size="5 5 0.1" material="floor"/>{''.join(geoms)}
</worldbody></mujoco>"""
    return xml, files


def render(model_id: str, azimuth: float, as_obj: bool = False) -> np.ndarray:
    xml, files = scene_xml(model_id, as_obj)
    m = mujoco.MjModel.from_xml_string(xml, files)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    rec = json.loads((FURN / f"{model_id}.json").read_text(encoding="utf-8"))
    sx, sy, sz = rec["size_m"]
    cam.lookat[:] = (0, 0, sz * 0.45)
    cam.distance = max(sx, sy, sz) * 1.9 + 0.3
    cam.azimuth, cam.elevation = azimuth, -22
    with mujoco.Renderer(m, TILE, TILE) as r:
        r.update_scene(d, cam)
        return r.render()


def tile(model_id: str) -> Image.Image:
    rec = json.loads((FURN / f"{model_id}.json").read_text(encoding="utf-8"))
    out = Image.new("RGB", (2 * TILE, TILE + LABEL), "white")
    for i, az in enumerate((-60, 120)):  # front three-quarter (glTF models face +y: the viewer), back
        out.paste(Image.fromarray(render(model_id, az)), (i * TILE, LABEL))
    faces = sum(p["faces"] for p in rec["parts"])
    size = " x ".join(f"{s:.2f}" for s in rec["size_m"])
    ImageDraw.Draw(out).text((8, 6), f"{model_id}   {faces} faces   {size} m   (front / back)", fill="black")
    return out


def main() -> int:
    args = sys.argv[1:]
    parity = "--parity" in args
    ids = [a for a in args if a != "--parity"] or sorted(p.stem for p in FURN.glob("*.json")
                                                          if p.name != "MANIFEST.json")
    if parity:  # pixels off by more than 2 levels: at most 0.01% (shadow-edge rounding of OBJ text)
        worst = 0.0
        for mid in ids:
            share = max(float(np.mean(np.abs(render(mid, az).astype(int)
                                              - render(mid, az, as_obj=True).astype(int)).max(2) > 2))
                        for az in (-60, 120))
            worst = max(worst, share)
            print(f"{mid:26s} pixels differing .msh vs OBJ: {share * 100:.4f}%")
        return 0 if worst <= 1e-4 else 1
    cols = 2
    tiles = [tile(mid) for mid in ids]
    rows = (len(tiles) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * 2 * TILE, rows * (TILE + LABEL)), "white")
    for i, t in enumerate(tiles):
        sheet.paste(t, ((i % cols) * 2 * TILE, (i // cols) * (TILE + LABEL)))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(OUT)
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
