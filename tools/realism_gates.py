"""Measurable realism gates on the fixed-pose renders of tools/realism_shots.py (development tool):
each pixel is attributed to its geom by a segmentation render, so the measures are exact.

  G1  ceiling: mean luminance 165 to 230, never darker than the upper walls in the same shot
  G2  clipped white (all channels 254 or more): under 0.3% of the frame, none outside a light fitting
  G3  sky (window glass): mean luminance 225 or more and brighter than every interior wall
  G4  bare wall texture: standard deviation of luminance 4 to 9 (within a wall, per shot)
  and per shot the mean luminance of walls, floor, and ceiling (for tuning the light balance).

    .venv\\Scripts\\python tools\\realism_gates.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from realism_shots import FOVY, H, SHOTS, W, light_eye  # noqa: E402
from robot_env.system import RobotSystem  # noqa: E402


def lum(rgb: np.ndarray) -> np.ndarray:
    return rgb[..., 0] * 0.2126 + rgb[..., 1] * 0.7152 + rgb[..., 2] * 0.0722


def classify(m) -> dict[str, np.ndarray]:
    """Geom ids by role, from names and materials."""
    roles = {"wall": [], "ceiling": [], "floor": [], "glass": [], "fitting": []}
    for g in range(m.ngeom):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        mat = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MATERIAL, int(m.geom_matid[g])) if m.geom_matid[g] >= 0 else ""
        if mat in ("light_panel", "fitting_frame"):
            roles["fitting"].append(g)
        elif name == "ceiling" or mat == "ceiling":
            roles["ceiling"].append(g)
        elif mat == "glass" or mat.startswith("view_"):
            roles["glass"].append(g)
        elif name.startswith("floor"):
            roles["floor"].append(g)
        elif mat == "plaster":
            roles["wall"].append(g)
    return {k: np.array(v, dtype=int) for k, v in roles.items()}


def measure() -> list[dict]:
    s = RobotSystem(cats=0)
    s.reset(4.4, -4.4, math.pi / 4, (4.0, 4.0))
    m, d = s.sim.model, s.sim.data
    m.vis.global_.fovy = FOVY
    roles = classify(m)
    r = mujoco.Renderer(m, height=H, width=W)
    opt = mujoco.MjvOption()
    opt.geomgroup[3] = 1
    lights = s.sim.lights
    m.geom_rgba[[m.geom(n).id for n in ("goal_disc", "goal_pole", "goal_flag")], 3] = 0.0  # no task marker
    rows = []
    for name, (eye, at, *fov) in SHOTS.items():
        m.vis.global_.fovy = fov[0] if fov else FOVY
        eye, at = np.array(eye, dtype=float), np.array(at, dtype=float)
        v = at - eye
        cam = mujoco.MjvCamera()
        cam.lookat[:] = at
        cam.distance = float(np.linalg.norm(v))
        cam.azimuth = math.degrees(math.atan2(v[1], v[0]))
        cam.elevation = math.degrees(math.asin(v[2] / np.linalg.norm(v)))
        light_eye(m, lights, eye, at)
        s.sim.before_render()
        r.update_scene(d, cam, opt)
        r.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 1
        r.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 1
        rgb = r.render().astype(np.float64)
        r.enable_segmentation_rendering()
        r.update_scene(d, cam, opt)
        seg = r.render()[..., 0]  # geom id per pixel (-1: none)
        r.disable_segmentation_rendering()
        L = lum(rgb)
        row = {"shot": name}
        for role, ids in roles.items():
            mask = np.isin(seg, ids)
            row[role] = (float(L[mask].mean()), float(L[mask].std()), int(mask.sum())) if mask.sum() > 200 else None
        # upper walls: wall pixels in the frame's top third
        top = np.isin(seg, roles["wall"]) & (np.arange(H)[:, None] < H // 3)
        row["upper_wall"] = float(L[top].mean()) if top.sum() > 200 else None
        clipped = (rgb >= 254).all(axis=2)
        row["clipped_pct"] = 100.0 * clipped.mean()
        row["clipped_outside_fitting_px"] = int((clipped & ~np.isin(seg, roles["fitting"])).sum())
        rows.append(row)
    r.close()
    s.close()
    return rows


def report(rows: list[dict]) -> bool:
    ok = True
    fmt = lambda v: "-" if v is None else f"{v[0]:.0f}/{v[1]:.1f}"  # mean/std
    print(f"{'shot':24s} {'wall':>10s} {'upper':>6s} {'floor':>10s} {'ceiling':>10s} {'glass':>10s} {'clip%':>6s} {'out':>5s}")
    for r in rows:
        up = "-" if r["upper_wall"] is None else f"{r['upper_wall']:.0f}"
        print(f"{r['shot']:24s} {fmt(r['wall']):>10s} {up:>6s} {fmt(r['floor']):>10s} {fmt(r['ceiling']):>10s} "
              f"{fmt(r['glass']):>10s} {r['clipped_pct']:6.2f} {r['clipped_outside_fitting_px']:5d}")
        if r["ceiling"] is not None:
            ok &= 165 <= r["ceiling"][0] <= 230 and (r["upper_wall"] is None or r["ceiling"][0] >= r["upper_wall"])
        ok &= r["clipped_pct"] < 0.3 and r["clipped_outside_fitting_px"] == 0
        if r["glass"] is not None:
            ok &= r["glass"][0] >= 225 and (r["wall"] is None or r["glass"][0] > r["wall"][0])
        if r["wall"] is not None:
            ok &= 4 <= r["wall"][1] <= 9
    print("gates G1 to G4:", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if report(measure()) else 1)
