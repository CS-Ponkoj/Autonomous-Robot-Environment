"""Screenshots of the realistic cats in the world (chase view and robot camera), with each cat
placed in the open corridor in front of the robot. Development check.

    .venv\\Scripts\\python tools\\cat_shots.py          # writes qa_output/cats/world_*.png
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from robot_env.app import VIEWS, App  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "qa_output" / "cats"


def shot(name: str, view: str, robot=(-4.0, 0.0, 0.0), cats=(), frames: int = 30, distance: float = 1.6) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"world_{name}.png"
    app = App(1000, path, frames, None, VIEWS.index(view), cats=max(3, len(cats)), cat_seed=16)
    app.system.reset(*robot, (4.5, 0.0))
    herd = app.system.cats
    for i in range(len(herd.cats)):  # cats not shown are parked far down the corridor's other end
        herd.place(i, 4.2 - 0.6 * i, -0.3 if i % 2 else 0.3, math.pi, state="pause")
    for i, (x, y, yaw) in enumerate(cats):
        herd.place(i, x, y, yaw, state="pause")
    app.view.reset()
    app.view.distance = distance
    app.hud = "none"
    app.run()
    return path


if __name__ == "__main__":
    print(shot("chase_three", "chase", cats=((-2.7, 0.25, math.pi * 0.9), (-2.4, -0.3, math.pi * 1.15), (-1.9, 0.05, math.pi))))
    print(shot("robot_camera_three", "robot camera", cats=((-2.9, 0.2, math.pi * 0.85), (-2.6, -0.25, math.pi * 1.2), (-2.2, 0.1, math.pi))))
    print(shot("close_tabby", "chase", cats=((-3.25, 0.0, math.pi * 0.75),), distance=0.9))
