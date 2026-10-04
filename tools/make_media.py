"""Record the README media: a demo video (MP4) and an animated GIF of the robot driving itself to
a goal in the real window, plus still screenshots. Development-only (needs requirements-dev.txt).

    .venv\\Scripts\\python -m pip install -r requirements-dev.txt
    .venv\\Scripts\\python tools\\make_media.py            # writes docs/demo.mp4, docs/demo.gif, docs/*.png
    .venv\\Scripts\\python tools\\make_media.py --gif-only # rebuilds docs/demo.gif from docs/demo.mp4

Pinned: goal seed 1000, speed level 2, 3 cats with cat seed 16, driven by the baseline driver.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pygame

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from robot_env import config as C  # noqa: E402
from robot_env.app import App  # noqa: E402
from robot_env.baseline import BaselineDriver  # noqa: E402

DOCS = Path(__file__).resolve().parent.parent / "docs"
SEED = 1000
LEVEL = 2  # 1-based speed level
CATS, CAT_SEED = 3, 16  # pinned: three wandering cats that cross the robot's route
# (seconds from start, view): chase with panels, the robot's own camera, chase again; the top
# view is shown once the goal is reached
VIEW_PLAN = ((0.0, 0), (6.0, 3), (10.0, 0))
TAIL = 3.0  # seconds kept after the goal is reached (top view)
GIF_WIDTH = 420
GIF_FPS = 8
GIF_COLORS = 64
VIDEO_FPS = 30


def record() -> list[tuple[float, np.ndarray]]:
    driver = BaselineDriver(C.SPEED_LEVELS[LEVEL - 1])
    app = App(SEED, screenshot=None, frames=60 * 60, view=0, driver=driver, speed_level=LEVEL - 1,
              cats=CATS, cat_seed=CAT_SEED)
    # Recording only: ignore window focus changes, so using the desktop does not pause the demo
    # (the simulator itself always stops when its window loses focus).
    pygame.event.set_blocked([pygame.WINDOWFOCUSLOST, pygame.WINDOWFOCUSGAINED])
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
        frames.append((t, pygame.surfarray.array3d(app.screen).swapaxes(0, 1).copy()))
        if state["end"] is not None and t >= state["end"]:
            app.frames_left = 1  # stop after this frame

    def update_episode():
        raw_update()
        if app.episode.status != "running" and state["end"] is None:
            state["end"] = app.system.time + TAIL

    app.draw, app.update_episode = draw, update_episode
    summary = app.run()
    print("recorded", len(frames), "frames;", {k: summary[k] for k in ("status", "episode_time", "collisions")})
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
    write_gif(resample(frames, GIF_FPS))
    for name in ("demo.mp4", "demo.gif"):
        print(name, f"{(DOCS / name).stat().st_size / 1e6:.1f} MB")


def write_gif(frames: list[np.ndarray]) -> None:
    """The small README GIF (GitHub shows it inline; kept under 5 MB)."""
    from PIL import Image
    small = []
    for f in frames:
        img = Image.fromarray(f)
        img = img.resize((GIF_WIDTH, round(GIF_WIDTH * img.height / img.width)), Image.LANCZOS)
        small.append(img.convert("P", palette=Image.ADAPTIVE, colors=GIF_COLORS))
    small[0].save(DOCS / "demo.gif", save_all=True, append_images=small[1:], duration=round(1000 / GIF_FPS),
                  loop=0, optimize=True)


def gif_from_video() -> None:
    """Rebuild the GIF from the recorded MP4 (no window needed)."""
    import imageio.v2 as imageio
    video = list(imageio.mimread(DOCS / "demo.mp4", memtest=False))
    step = VIDEO_FPS / GIF_FPS
    write_gif([video[int(round(i * step))] for i in range(int(len(video) / step))])
    print("demo.gif", f"{(DOCS / 'demo.gif').stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    if "--gif-only" in sys.argv[1:]:
        gif_from_video()
    else:
        write_media(record())
