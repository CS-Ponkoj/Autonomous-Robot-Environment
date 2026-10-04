"""Rendered performance evidence with a fixed protocol: repeated trials per view, driving with
scripted keys, after a fixed warm-up. Results go to qa_output/.

    .venv\\Scripts\\python tools\\fps_protocol.py                 # 3 trials per view
    .venv\\Scripts\\python tools\\fps_protocol.py --trials 5 --note "browser open"

Reports per view: FPS (median and minimum over trials), frame-time p50/p95/p99 and worst
frame, late frames (> 1.5 x the 1/60 s budget), and the real-time factor. Also records the host,
renderer, window and render size, power plan, and a free-text background-load note. A field
that cannot be read is reported as "unavailable", never guessed.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from robot_env.app import FPS, VIEWS, WINDOW, App, parse_script  # noqa: E402

WARMUP_FRAMES = 60
SAMPLE_FRAMES = 600
SCRIPT = "W:3;D:1;W:3;A:1;W:3"
OUT = Path(__file__).resolve().parent.parent / "qa_output"


def _run(cmd: list[str]) -> str:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout.strip()
        return out or "unavailable"
    except (OSError, subprocess.SubprocessError):
        return "unavailable"


def host_info() -> dict:
    info = {"platform": platform.platform(), "python": platform.python_version(),
            "machine": platform.machine() or "unavailable", "processor": platform.processor() or "unavailable"}
    if sys.platform == "win32":
        plan = _run(["powercfg", "/getactivescheme"])
        info["power_plan"] = plan.split("(")[-1].rstrip(")") if "(" in plan else plan
        gpu = _run(["powershell", "-NoProfile", "-Command",
                    "(Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name) -join '; '"])
        info["gpu"] = gpu
    else:
        info["power_plan"] = "unavailable"
        info["gpu"] = "unavailable"
    return info


def renderer_string(app) -> str:
    """The OpenGL renderer of the app's MuJoCo rendering context. Must be called after the app
    has rendered (the context exists and is current). PyOpenGL is optional."""
    try:
        app.renderer.update_scene(app.system.sim.data)
        app.renderer.render()  # makes the context current
        from OpenGL import GL  # optional; not a project dependency
        return GL.glGetString(GL.GL_RENDERER).decode()
    except ImportError:
        return "unavailable (PyOpenGL not installed)"
    except Exception as e:
        return f"unavailable ({type(e).__name__})"


def trial(view: int) -> dict:
    app = App(1000, None, WARMUP_FRAMES + SAMPLE_FRAMES, parse_script(SCRIPT), view)
    renderer = renderer_string(app)
    stamps: list[float] = []
    raw = app.draw

    def draw():
        raw()
        stamps.append(time.perf_counter())
    app.draw = draw
    summary = app.run()
    frames = [b - a for a, b in zip(stamps[WARMUP_FRAMES:], stamps[WARMUP_FRAMES + 1:])]
    frames.sort()

    def pct(q):
        return 1000 * frames[min(int(q * len(frames)), len(frames) - 1)]
    return {"fps": len(frames) / sum(frames), "p50_ms": pct(0.50), "p95_ms": pct(0.95), "p99_ms": pct(0.99),
            "worst_ms": 1000 * frames[-1], "late_frames": sum(f > 1.5 / FPS for f in frames),
            "frames": len(frames), "rtf": summary["real_time_factor"], "renderer": renderer}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--trials", type=int, default=3)
    p.add_argument("--note", default="", help="background load or other conditions during the run")
    p.add_argument("--out", type=Path, default=OUT / "fps_protocol.json")
    a = p.parse_args(argv)
    if a.trials < 3:
        p.error("at least 3 trials per view")
    result = {"protocol": 1, "command": "python tools/fps_protocol.py " + " ".join(sys.argv[1:]),
              "time": time.strftime("%Y-%m-%d %H:%M:%S"), "warmup_frames": WARMUP_FRAMES,
              "sample_frames": SAMPLE_FRAMES, "script": SCRIPT, "window": list(WINDOW), "render_size": list(WINDOW),
              "frame_budget_ms": 1000 / FPS, "renderer": None, "host": host_info(),
              "background_note": a.note or "unavailable", "views": {}}
    for view, name in enumerate(VIEWS):
        trials = [trial(view) for _ in range(a.trials)]
        names = {t["renderer"] for t in trials} | ({result["renderer"]} if result["renderer"] else set())
        if len(names) != 1:
            raise SystemExit(f"trials used different renderers: {sorted(names)}")
        result["renderer"] = names.pop()
        fps = [t["fps"] for t in trials]
        result["views"][name] = {
            "fps_median": statistics.median(fps), "fps_min": min(fps),
            "p50_ms": max(t["p50_ms"] for t in trials), "p95_ms": max(t["p95_ms"] for t in trials),
            "p99_ms": max(t["p99_ms"] for t in trials), "worst_ms": max(t["worst_ms"] for t in trials),
            "late_frames": sum(t["late_frames"] for t in trials), "rtf_min": min(t["rtf"] for t in trials),
            "trials": trials}
        v = result["views"][name]
        print(f"{name:13s} fps median {v['fps_median']:.1f} min {v['fps_min']:.1f}  p50 {v['p50_ms']:.1f} "
              f"p95 {v['p95_ms']:.1f} p99 {v['p99_ms']:.1f} worst {v['worst_ms']:.1f} ms  late {v['late_frames']}  "
              f"rtf min {v['rtf_min']:.3f}", flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
