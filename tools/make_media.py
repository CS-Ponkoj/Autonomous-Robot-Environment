"""Record the README media in the real window: the demo video (MP4) and GIF of the robot driving
itself to a goal, a GIF of a cat giving way in a doorway, a time-lapse GIF of the cats roaming the
floor, and still screenshots. Development-only (needs requirements-dev.txt).

    .venv\\Scripts\\python -m pip install -r requirements-dev.txt
    .venv\\Scripts\\python tools\\make_media.py            # writes everything under docs/
    .venv\\Scripts\\python tools\\make_media.py --gif-only # rebuilds docs/demo.gif from docs/demo.mp4

Pinned: goal seed 1000, speed level 2, 4 cats (the window's default) with cat seed 16 (demo,
time-lapse, and scene stills); the doorway scene has 4 cats too (cat seed 2): one resting in the
doorway, the others sitting far away, and the robot driven straight ahead by a key script. Every
image in the README comes from this tool, and goal-labelled images are refused unless the goal
was reached with no collision.
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
CATS, CAT_SEED = 4, 16  # pinned: the window's four cats
# (seconds from start, view): chase with panels, the robot's own camera, chase again; the top
# view is shown once the goal is reached
VIEW_PLAN = ((0.0, 0), (6.0, 3), (10.0, 0))
TAIL = 3.0  # seconds kept after the goal is reached (top view)
GIF_WIDTH = 420
GIF_FPS = 8
GIF_COLORS = 64
VIDEO_FPS = 30
DOOR = (2.5, 0.75)  # the lab doorway on the corridor's north side
OUT_OF_THE_WAY = ((-3.6, 3.3, 0.5), (-3.2, -3.4, 2.0), (3.4, -3.0, -2.4))  # office, storage, reception
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
    if summary["status"] != "success" or summary["collisions"]:
        raise RuntimeError(f"the demo did not reach the goal cleanly: {summary['status']}, {summary['collisions']} collisions")
    return frames


def record_doorway() -> list[tuple[float, np.ndarray]]:
    """A cat resting in the lab doorway, seen by the robot's own camera; the robot drives straight
    at it (W held). The cat steps out of the robot's path and the robot drives through: no push,
    no contact."""
    app = _quiet(App(SEED, screenshot=None, frames=60 * 40, script=parse_script("none:1.0;W:12"), view=VIEWS.index("robot camera"),
                     speed_level=LEVEL - 1, cats=CATS, cat_seed=2))
    s = app.system
    s.reset(DOOR[0], DOOR[1] - 1.2, math.pi / 2, (DOOR[0], 3.6))
    s.cats.place(0, DOOR[0], DOOR[1] + 0.1, math.pi / 2, state="pause")
    for i, (x, y, yaw) in enumerate(OUT_OF_THE_WAY[:CATS - 1], start=1):  # the others rest far from the door
        s.cats.place(i, x, y, yaw, state="sit")
    app.view.reset()
    app.view.distance, app.view.elevation = 2.0, -24.0  # (if switched to the chase view)
    frames: list[tuple[float, np.ndarray]] = []
    raw_draw = app.draw

    def draw():
        raw_draw()
        frames.append((s.time, _frame(app)))
        if s.time > 11.0 or s.sim.true_pose()[1] > DOOR[1] + 0.9:
            app.frames_left = 1

    app.draw = draw
    summary = app.run()
    print("doorway: recorded", len(frames), "frames;", {k: summary[k] for k in ("collisions",)},
          "cat contacts", s.cat_contacts)
    if summary["collisions"] or s.cat_contacts:
        raise RuntimeError("the doorway scene had a collision or a cat contact")
    return frames


def record_roaming() -> list[tuple[float, np.ndarray]]:
    """Top view while the robot stays parked: four cats roam the whole floor."""
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


def _scene(robot, view: int, hud: str = "compact", frames: int = 150, setup=None, distance=None,
           elevation=None, driver=None, until_goal: bool = False) -> np.ndarray:
    """One still: the robot placed at `robot` (x, y, yaw) in the window, the given view and panels,
    after `frames` frames (the camera settles, the cats move); `setup(app)` may place the cats."""
    app = _quiet(App(SEED, screenshot=None, frames=None, view=view, driver=driver, speed_level=LEVEL - 1,
                     cats=CATS, cat_seed=CAT_SEED))
    try:
        app.hud = hud
        if robot is not None:
            app.system.reset(*robot, app.episode.task.goal)
        if setup is not None:
            setup(app)
        app.view.reset()
        if distance is not None:
            app.view.distance = distance
        if elevation is not None:
            app.view.elevation = elevation
        image = None
        for k in range(frames if not until_goal else 60 * 60):
            app.handle_events()
            app.simulate(1 / 60)
            app._frame_dt = 1 / 60
            app.draw()
            app.clock.tick(60)  # the panel's frame rate reads the window's 60 per second
            if until_goal and app.episode.status == "success":
                for _ in range(45):  # the success banner and a settled camera
                    app.simulate(1 / 60)
                    app.draw()
                    app.clock.tick(60)
                break
        if until_goal and (app.episode.status != "success" or app.system.collisions):
            raise RuntimeError(f"the goal was not reached cleanly: {app.episode.status}, "
                               f"{app.system.collisions} collisions")
        image = _frame(app)
    finally:
        app._close_all()
    return image


def _cats_ahead(app) -> None:
    """Four cats down the corridor in front of the robot (robot at its west end facing east),
    sitting or standing 1 to 3 m away, all within its camera's view; each placed clear of walls,
    furniture, and the others."""
    from robot_env.cats import CAT_GAP, WALL_GAP
    herd = app.system.cats
    poses = ((-3.05, 0.22, 3.0, "sit"), (-2.55, -0.28, 2.7, "pause"), (-1.95, 0.32, -2.8, "sit"),
             (-1.35, -0.12, 3.25, "pause"))
    for i, (x, y, yaw, state) in enumerate(poses[:herd.n]):
        herd.place(i, x, y, yaw, state=state)
    for c in herd.cats:
        wall, _, other = herd.gaps(c, c.x, c.y, c.yaw)
        if wall < WALL_GAP or other < CAT_GAP:
            raise RuntimeError(f"cat {c.index} placed too close (wall {wall:.3f} m, cat {other:.3f} m)")


def _cat_sitting(app) -> None:
    """A cat sitting across the corridor 0.9 m in front of the robot (robot at (-2.7, 0) facing
    east); the other cats sit far away."""
    herd = app.system.cats
    herd.place(0, -1.8, 0.0, math.pi / 2, state="sit")
    for i, (x, y, yaw) in enumerate(OUT_OF_THE_WAY[:herd.n - 1], start=1):
        herd.place(i, x, y, yaw, state="sit")


def write_scene_stills() -> None:
    """The scene screenshots in the README (each composed in the window, with four cats)."""
    shots = {
        "cat_sitting_robot_camera.png": dict(robot=(-2.7, 0.0, 0.0), view=VIEWS.index("robot camera"),
                                             setup=_cat_sitting, frames=180),
        "cat_sitting.png": dict(robot=(-2.7, 0.0, 0.0), view=0, setup=_cat_sitting, frames=180),
        "corridor.png": dict(robot=(-3.6, 0.0, 0.0), view=0),
        "reception.png": dict(robot=(1.0, -1.6, -0.75), view=0),
        "lab_robot_camera.png": dict(robot=(0.9, 1.6, 0.55), view=VIEWS.index("robot camera")),
        "robot_closeup.png": dict(robot=(-2.4, 2.6, 0.5), view=VIEWS.index("orbit"), hud="none", distance=0.7,
                                  elevation=-25.0),
        "floor_top.png": dict(robot=None, view=VIEWS.index("top"), hud="none"),
        "cats_robot_camera.png": dict(robot=(-4.2, 0.0, 0.0), view=VIEWS.index("robot camera"), setup=_cats_ahead,
                                      frames=150),  # the sitting ones have sat down
    }
    for name, shot in shots.items():
        _still(_scene(**shot), name)
    # goal reached with the full panels (H), driven by the baseline driver; and its lidar panel
    goal = _scene(None, 0, hud="full", driver=BaselineDriver(C.SPEED_LEVELS[LEVEL - 1]), until_goal=True)
    _still(goal, "screenshot_goal.png")
    h, w = goal.shape[:2]
    _still(goal[h - 16 - 220 - 44:h - 16 + 16, w - 16 - 220 - 30:w - 16 + 30], "lidar_panel.png")


def gif_from_video() -> None:
    """Rebuild the GIF from the recorded MP4 (no window needed)."""
    import imageio.v2 as imageio
    video = list(imageio.mimread(DOCS / "demo.mp4", memtest=False))
    step = VIDEO_FPS / GIF_FPS
    write_gif([video[int(round(i * step))] for i in range(int(len(video) / step))], DOCS / "demo.gif")


if __name__ == "__main__":
    if "--gif-only" in sys.argv[1:]:
        gif_from_video()
    elif "--stills-only" in sys.argv[1:]:
        write_scene_stills()
    else:
        demo = record()
        write_media(demo)
        write_stills(demo)
        write_gif(resample(record_doorway(), 10), DOCS / "doorway.gif", fps=10)
        roam = [f for _, f in record_roaming()]
        write_gif(roam, DOCS / "cats_roaming.gif", fps=12, width=560)
        write_scene_stills()
