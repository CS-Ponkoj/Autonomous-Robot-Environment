"""Measure the viewer camera's stability headlessly (no window): drive a scripted route at 60
frames per second and record, per frame, how far the camera moved, whether walls limited the
zoom, and how often the line-of-sight detour changed.

    .venv\\Scripts\\python tools\\camera_jitter.py            # seeds 1003 and 1007, chase and orbit, zoom 4 m
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from robot_env import config as C  # noqa: E402
from robot_env.app import VIEWS, ViewCamera  # noqa: E402
from robot_env.baseline import BaselineDriver  # noqa: E402
from robot_env.layout import RoomMap  # noqa: E402
from robot_env.system import RobotSystem  # noqa: E402

FRAME = 1 / 60


def camera_position(cam) -> np.ndarray:
    az, el = math.radians(cam.azimuth), math.radians(cam.elevation)
    back = np.array([-math.cos(el) * math.cos(az), -math.cos(el) * math.sin(az), -math.sin(el)])
    return np.asarray(cam.lookat) + back * cam.distance


def robot_hidden(sim, cam_view, pos) -> bool:
    """True when furniture or a wall hides the robot's top centre from the camera point."""
    x, y, _ = sim.true_pose()
    target = np.array([x, y, 0.16])
    d = target - pos
    n = float(np.linalg.norm(d))
    hit = sim.ray_to_view_blocker(pos, d / n)
    return 0 <= hit < n - 0.03


def measure(seed: int, view: str, zoom: float = 4.0, seconds: float = 40.0, still: float = 0.0,
            shots: Path | None = None) -> dict:
    """Baseline drive to the goal of `seed` (or stand still for `still` seconds), one camera
    update per 1/60 s of simulated time."""
    s = RobotSystem(cats=0)
    task = RoomMap(s.sim.model).sample_task(seed)
    s.reset(*task.start, task.goal)
    cam = ViewCamera()
    cam.mode = VIEWS.index(view)
    cam.distance = zoom
    d = BaselineDriver()
    jumps, limited, detours, dists, hidden = [], 0, 0, [], 0
    closest = (np.inf, None)
    prev_pos, prev_detour = None, None
    frames = int(round((still or seconds) / FRAME))
    next_decision = 0.0
    for _ in range(frames):
        if not still and s.time >= next_decision - 1e-9:
            s.apply(d.decide(s.observe()))
            next_decision = s.time + C.DECISION_PERIOD
        s.advance(FRAME)
        cam.apply(s.sim, FRAME)
        pos = camera_position(cam.cam)
        if prev_pos is not None:
            jumps.append(float(np.linalg.norm(pos - prev_pos)))
            detours += cam._detour != prev_detour
        prev_pos, prev_detour = pos, cam._detour
        limited += cam.limited
        dists.append(cam.cam.distance)
        hidden += robot_hidden(s.sim, cam, pos)
        if shots is not None and cam.cam.distance < closest[0]:
            c = cam.cam
            closest = (c.distance, (s.sim.data.qpos.copy(), np.array(c.lookat), c.azimuth, c.elevation, c.distance,
                                    cam.fovy))
        if not still and s.observe().goal_distance <= C.GOAL_RADIUS:
            break
    if shots is not None and closest[1] is not None:
        shots.mkdir(parents=True, exist_ok=True)
        from PIL import Image
        Image.fromarray(_snapshot(s, *closest[1])).save(shots / f"closest_{seed}_{view}.png")
    s.close()
    j = np.array(jumps)
    return {"seed": seed, "view": view, "frames": len(j) + 1, "limited_share": round(limited / (len(j) + 1), 3),
            "detour_changes": detours, "mean_distance_m": round(float(np.mean(dists)), 2),
            "min_distance_m": round(float(np.min(dists)), 3), "robot_hidden_frames": hidden, "max_jump_m": round(float(j.max()), 4),
            "p99_jump_m": round(float(np.percentile(j, 99)), 4), "settled_last_s_max_m": round(float(j[-60:].max()), 5)}


def _snapshot(s, qpos, lookat, azimuth, elevation, distance, fovy) -> np.ndarray:
    """Render the chase/orbit camera at a saved state (after the run)."""
    import mujoco
    s.sim.data.qpos[:] = qpos
    mujoco.mj_forward(s.sim.model, s.sim.data)
    cam = mujoco.MjvCamera()
    cam.lookat[:], cam.azimuth, cam.elevation, cam.distance = lookat, azimuth, elevation, distance
    r = mujoco.Renderer(s.sim.model, 360, 640)
    s.sim.model.vis.global_.fovy = fovy
    r.update_scene(s.sim.data, camera=cam, scene_option=mujoco.MjvOption())
    image = r.render().copy()
    r.close()
    return image


if __name__ == "__main__":
    out = Path(__file__).resolve().parent.parent / "qa_output" / "camera"
    for seed in (1003, 1007):
        for view in ("chase", "orbit"):
            print(measure(seed, view, shots=out))
