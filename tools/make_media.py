"""Record the README media in the real window: the demo video (MP4) and GIF of the robot driving
itself to a goal, a GIF of a cat giving way in a doorway, a time-lapse GIF of the cats roaming the
floor, and still screenshots. Development-only (needs requirements-dev.txt).

    .venv\\Scripts\\python -m pip install -r requirements-dev.txt
    .venv\\Scripts\\python tools\\make_media.py            # writes everything under docs/
    .venv\\Scripts\\python tools\\make_media.py --gif-only # rebuilds docs/demo.gif from docs/demo.mp4

Pinned: goal seed 1000, speed level 2, 3 cats with cat seed 16 (demo and time-lapse); the doorway
scene uses one cat (cat seed 2) and the robot driven straight ahead by a key script.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pygame

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from robot_env import config as C  # noqa: E402
from robot_env.app import VIEWS, App, parse_script  # noqa: E402
from robot_env.baseline import BaselineDriver  # noqa: E402

DOCS = Path(__file__).resolve().parent.parent / "docs"
SEED = 1000
LEVEL = 2  # 1-based speed level
CATS, CAT_SEED = 3, 16  # pinned: three cats that cross the robot's route
# (seconds from start, view): chase with panels, the robot's own camera, chase again; the top
# view is shown once the goal is reached
VIEW_PLAN = ((0.0, 0), (6.0, 3), (10.0, 0))
TAIL = 3.0  # seconds kept after the goal is reached (top view)
GIF_WIDTH = 420
GIF_FPS = 8
GIF_COLORS = 64
VIDEO_FPS = 30
DOOR = (2.5, 0.75)  # the lab doorway on the corridor's north side
ROAM_SECONDS = 90.0  # simulated time in the roaming time-lapse
ROAM_SAMPLE = 0.5  # s of simulated time between time-lapse frames (shown at 12 per second: 6x)


def _quiet(app: App) -> App:
    # Recording only: ignore window focus changes, so using the desktop does not pause the
    # recording (the simulator itself always stops when its window loses focus).
    pygame.event.set_blocked([pygame.WINDOWFOCUSLOST, pygame.WINDOWFOCUSGAINED])
    return app


def _frame(app: App) -> np.ndarray:
    return pygame.surfarray.array3d(app.screen).swapaxes(0, 1).copy()


def record() -> list[tuple[float, np.ndarray]]:
    """The demo: the baseline driver takes the robot from the office to the lab past the cats."""
    driver = BaselineDriver(C.SPEED_LEVELS[LEVEL - 1])
    app = _quiet(App(SEED, screenshot=None, frames=60 * 60, view=0, driver=driver, speed_level=LEVEL - 1,
                     cats=CATS, cat_seed=CAT_SEED))
    frames: list[tuple[float, np.ndarray]] = []  # (simulated time, image)
    raw_draw, raw_update = app.draw, app.update_episode
    state = {"end": None}

    def draw():
        t = app.system.time
        view = 1 if state["end"] is not None else [v for start, v in VIEW_PLAN if t >= start][-1]
        if view != app.view.mode:
            app.view.mode = view
            app.view.reset()
        raw_draw()
        frames.append((t, _frame(app)))
        if state["end"] is not None and t >= state["end"]:
            app.frames_left = 1  # stop after this frame

    def update_episode():
        raw_update()
        if app.episode.status != "running" and state["end"] is None:
            state["end"] = app.system.time + TAIL

    app.draw, app.update_episode = draw, update_episode
    summary = app.run()
    print("demo: recorded", len(frames), "frames;", {k: summary[k] for k in ("status", "episode_time", "collisions")})
    return frames


def record_doorway() -> list[tuple[float, np.ndarray]]:
    """A cat resting in the lab doorway; the robot drives straight at it (W held). The cat steps
    out of the robot's path and the robot drives through: no push, no contact."""
    app = _quiet(App(SEED, screenshot=None, frames=60 * 40, script=parse_script("none:1.0;W:12"), view=0,
                     speed_level=LEVEL - 1, cats=1, cat_seed=2))
    s = app.system
    s.reset(DOOR[0], DOOR[1] - 1.2, math.pi / 2, (DOOR[0], 3.6))
    s.cats.place(0, DOOR[0], DOOR[1] + 0.1, math.pi / 2, state="pause")
    app.view.reset()
    app.view.distance, app.view.elevation = 2.0, -24.0
    frames: list[tuple[float, np.ndarray]] = []
    raw_draw = app.draw

    def draw():
        raw_draw()
        frames.append((s.time, _frame(app)))
        if s.time > 11.0 or s.sim.true_pose()[1] > DOOR[1] + 1.6:
            app.frames_left = 1

    app.draw = draw
    summary = app.run()
    print("doorway: recorded", len(frames), "frames;", {k: summary[k] for k in ("collisions",)},
          "cat contacts", s.cat_contacts)
    return frames


def record_roaming() -> list[tuple[float, np.ndarray]]:
    """Top view while the robot stays parked: three cats roam the whole floor."""
    app = _quiet(App(SEED, screenshot=None, frames=None, view=VIEWS.index("top"), cats=CATS, cat_seed=CAT_SEED))
    app.hud = "none"
    frames: list[tuple[float, np.ndarray]] = []
    raw_draw = app.draw
    state = {"next": 0.0}

    def draw():
        raw_draw()
        t = app.system.time
        if t >= state["next"]:
            frames.append((t, _frame(app)))
            state["next"] += ROAM_SAMPLE
        if t >= ROAM_SECONDS:
            app.frames_left = 1

    app.draw = draw
    app.frames_left = 10 ** 9
    app.run()
    visited = [sorted(c.visited) for c in app.system.cats.cats]
    print("roaming: recorded", len(frames), "frames; rooms visited per cat:", visited)
    return frames


def resample(frames, fps: float) -> list[np.ndarray]:
    """Frames at a fixed rate in SIMULATED time, so the media plays at real speed even though
    capturing slows the window down."""
    times = np.array([t for t, _ in frames])
    out = []
    for target in np.arange(times[0], times[-1], 1.0 / fps):
        out.append(frames[int(np.argmin(np.abs(times - target)))][1])
    return out


def write_media(frames) -> None:
    import imageio.v2 as imageio
    DOCS.mkdir(exist_ok=True)
    video = resample(frames, VIDEO_FPS)
    imageio.mimwrite(DOCS / "demo.mp4", video, fps=VIDEO_FPS, codec="libx264", quality=7, macro_block_size=8, pixelformat="yuv420p",
                     ffmpeg_params=["-movflags", "+faststart"])
    write_gif(resample(frames, GIF_FPS), DOCS / "demo.gif")
    for name in ("demo.mp4", "demo.gif"):
        print(name, f"{(DOCS / name).stat().st_size / 1e6:.1f} MB")


def write_gif(frames: list[np.ndarray], path: Path, fps: float = GIF_FPS, width: int = GIF_WIDTH,
              colors: int = GIF_COLORS) -> None:
    """A small GIF (GitHub shows it inline; each kept under 5 MB)."""
    from PIL import Image
    small = []
    for f in frames:
        img = Image.fromarray(f)
        img = img.resize((width, round(width * img.height / img.width)), Image.LANCZOS)
        small.append(img.convert("P", palette=Image.ADAPTIVE, colors=colors))
    small[0].save(path, save_all=True, append_images=small[1:], duration=round(1000 / fps), loop=0, optimize=True)
    print(path.name, f"{path.stat().st_size / 1e6:.1f} MB")


def _still(frame: np.ndarray, name: str) -> None:
    from PIL import Image
    Image.fromarray(frame).save(DOCS / name, optimize=True)
    print(name)


def write_stills(demo) -> None:
    """Screenshots for the README, taken from the recordings (exactly the published revision)."""
    times = np.array([t for t, _ in demo])
    _still(demo[int(np.argmin(np.abs(times - 8.0)))][1], "robot_camera_cat.png")  # the robot camera segment
    _still(demo[-1][1], "top_view_cats.png")  # goal reached, top view
    import importlib
    faces = importlib.import_module("tools.cat_face")
    faces.OUT = DOCS / "cat_faces.png"
    faces.main()


def gif_from_video() -> None:
    """Rebuild the GIF from the recorded MP4 (no window needed)."""
    import imageio.v2 as imageio
    video = list(imageio.mimread(DOCS / "demo.mp4", memtest=False))
    step = VIDEO_FPS / GIF_FPS
    write_gif([video[int(round(i * step))] for i in range(int(len(video) / step))], DOCS / "demo.gif")


if __name__ == "__main__":
    if "--gif-only" in sys.argv[1:]:
        gif_from_video()
    else:
        demo = record()
        write_media(demo)
        write_stills(demo)
        write_gif(resample(record_doorway(), 10), DOCS / "doorway.gif", fps=10)
        roam = [f for _, f in record_roaming()]
        write_gif(roam, DOCS / "cats_roaming.gif", fps=12, width=560)
