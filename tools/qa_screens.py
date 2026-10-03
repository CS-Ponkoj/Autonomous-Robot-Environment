"""QA screenshots of the office floor from fixed poses (saved to qa_output).

    .venv\Scripts\python tools\qa_screens.py
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import mujoco  # noqa: E402
import pygame  # noqa: E402

from robot_env.app import App  # noqa: E402
from robot_env.sim import RobotSim  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "qa_output"

SHOTS = [  # name, (x, y, yaw degrees), view, azimuth offset, elevation, distance, panels
    ("floor_top", (0.0, 0.0, 0), 1, 0, -22, 1.6, False),
    ("corridor_chase", (-3.8, 0.0, 0), 0, 0, -12, 1.6, False),
    ("doorway_chase", (-2.5, 0.2, 90), 0, 0, -18, 1.4, False),
    ("office_orbit", (-2.4, 2.4, 140), 2, 225, -35, 3.2, False),
    ("lab_robotcam", (0.6, 1.4, 35), 3, 0, -22, 1.6, False),
    ("reception_orbit", (2.0, -1.6, -60), 2, 135, -30, 3.0, False),
    ("storage_robotcam", (-3.0, -1.4, -90), 3, 0, -22, 1.6, False),
    ("robot_closeup", (0.0, 0.0, 30), 2, 200, -24, 0.55, False),
    ("panels_corridor", (1.0, 0.0, 180), 0, 0, -22, 1.6, True),
    ("window_robotcam", (3.6, 3.3, 0), 3, 0, -22, 1.6, False),
    ("office_window_orbit", (-1.6, 3.4, 90), 2, 90, -6, 2.4, False),
    ("robot_front3q", (0.0, 0.0, 0), 2, 210, -18, 0.6, False),
    ("robot_side", (0.0, 0.0, 0), 2, 90, -4, 0.55, False),
    ("robot_rear", (0.0, 0.0, 0), 2, 0, -14, 0.55, False),
    ("corridor_art_chase", (0.2, 0.0, 0), 0, 30, -10, 1.8, False),
]


def signs_check() -> None:
    """All four room signs from eye height (they must read left to right)."""
    sim = RobotSim()
    r = mujoco.Renderer(sim.model, 360, 960)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    out = pygame.Surface((960, 360 * 4))
    signs = (((-2.2, -0.7), 270), ((2.8, -0.7), 270), ((-1.7, 0.7), 90), ((3.3, 0.7), 90))
    for i, ((x, y), az) in enumerate(signs):
        cam.lookat[:] = (x, y, 1.6)
        cam.distance, cam.azimuth, cam.elevation = 1.0, az, 0
        r.update_scene(sim.data, camera=cam)
        img = r.render()
        out.blit(pygame.image.frombuffer(img.tobytes(), (960, 360), "RGB"), (0, 360 * i))
    pygame.image.save(pygame.transform.smoothscale(out, (480, 720)), str(OUT / "signs_check.png"))
    r.close()
    sim.close()


DETAILS = [  # name, look-at point, azimuth (view direction), elevation, distance
    ("chair", (-3.8, 3.7, 0.5), 135, -18, 1.6),
    ("desk", (-3.8, 4.4, 0.9), 90, -15, 1.7),
    ("sofa", (1.6, -4.4, 0.45), 270, -15, 2.4),
    ("door_hardware", (-2.95, 1.2, 1.0), 200, -5, 1.1),
    ("plant", (-0.45, 1.25, 0.5), 315, -15, 1.3),
    ("lab_bench", (2.6, 4.4, 1.0), 90, -20, 1.8),
]


def detail_shots() -> None:
    """Close-ups of furniture and fittings, two per row, for visual review."""
    sim = RobotSim()
    sim.reset(0.0, 0.0, 0.0, (4.0, -4.0))
    r = mujoco.Renderer(sim.model, 360, 640)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    out = pygame.Surface((1280, 360 * ((len(DETAILS) + 1) // 2)))
    for i, (_, look, az, el, dist) in enumerate(DETAILS):
        cam.lookat[:] = look
        cam.azimuth, cam.elevation, cam.distance = az, el, dist
        r.update_scene(sim.data, camera=cam)
        img = r.render()
        out.blit(pygame.image.frombuffer(img.tobytes(), (640, 360), "RGB"), (640 * (i % 2), 360 * (i // 2)))
    pygame.image.save(out, str(OUT / "details_check.png"))
    r.close()
    sim.close()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    signs_check()
    detail_shots()
    for name, (x, y, yaw), view, az, el, dist, panels in SHOTS:
        app = App(1000, screenshot=OUT / f"shot_{name}.png", frames=8, view=view)
        app.system.reset(x, y, math.radians(yaw), (4.0, -4.0))
        app.view.azimuth, app.view.elevation, app.view.distance = az, el, dist
        app.show_help = panels
        r = app.run()
        print(f"{name}: fps {r['fps']:.0f}")


if __name__ == "__main__":
    main()
