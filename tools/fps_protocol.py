"""Rendered performance evidence with a fixed protocol: repeated trials per view, driving with
scripted keys, after a fixed warm-up. Results go to qa_output/.

    .venv\\Scripts\\python tools\\fps_protocol.py                 # 3 trials per view
    .venv\\Scripts\\python tools\\fps_protocol.py --trials 5 --note "browser open"
    .venv\\Scripts\\python tools\\fps_protocol.py --gate          # exit 1 unless every view passes

Reports per view: FPS (median and minimum over trials), frame-time p50/p95/p99 and worst
frame, late frames (> 1.5 x the 1/60 s budget), and the real-time factor. Also records the host,
renderer, window and render size, power plan, the source revision (commit, and whether the
working tree differs from it), the cat count, the cats' kernels (compiled or not, first compile
with an empty cache, and loading from the cache), and a free-text background-load note. A field that
cannot be read is reported as "unavailable", never guessed.

The frame-rate gate (--gate, which is also --strict: a clean, known build that does not change
during the run; 4 cats, every view, every trial must pass): at least 60 FPS (the
app's 60 Hz schedule: at least 59.5 measured in each trial), p95 frame at most 20 ms, no frame
over 50 ms, and a real-time factor of at least 0.99.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from robot_env.app import FPS, VIEWS, WINDOW, App, parse_script  # noqa: E402
from robot_env.display import VSYNC  # noqa: E402

WARMUP_FRAMES = 60
SAMPLE_FRAMES = 600
SCRIPT = "W:3;D:1;W:3;A:1;W:3"
OUT = Path(__file__).resolve().parent.parent / "qa_output"
GATE = {"fps_min": 59.5, "p95_ms_max": 20.0, "worst_ms_max": 50.0, "rtf_min": 0.99, "cats": 4}


def _run(cmd: list[str]) -> str:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout.strip()
        return out or "unavailable"
    except (OSError, subprocess.SubprocessError):
        return "unavailable"


def kernel_times() -> dict:
    """The cats' compiled kernels: whether they are compiled, the first compile with an empty
    cache (measured in a separate process), and loading them from the cache here (before any
    trial, so no measured frame pays for it)."""
    import tempfile

    from robot_env import kernels
    cold = "unavailable"
    with tempfile.TemporaryDirectory() as cache:
        r = subprocess.run([sys.executable, "-c", "from robot_env import kernels; print(kernels.warm())"],
                           capture_output=True, text=True, timeout=600, cwd=Path(__file__).resolve().parent.parent,
                           env={**os.environ, "NUMBA_CACHE_DIR": cache})
        if r.returncode == 0 and r.stdout.strip():
            cold = round(float(r.stdout.strip().splitlines()[-1]), 2)
    return {"compiled": kernels.COMPILED, "cold_compile_s": cold, "cached_warm_s": round(kernels.warm(), 2)}


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


def sample_intervals(frame_times: list[float]) -> list[float]:
    """The measured intervals: after the warm-up ones (frame_times[i] is the interval ending at
    presented frame i + 1), exactly SAMPLE_FRAMES of them."""
    out = frame_times[WARMUP_FRAMES:WARMUP_FRAMES + SAMPLE_FRAMES]
    if len(out) != SAMPLE_FRAMES:
        raise RuntimeError(f"expected {SAMPLE_FRAMES} measured intervals, got {len(out)}")
    return out


def trial(view: int, cats: int) -> dict:
    # one frame more than warm-up plus sample: SAMPLE_FRAMES intervals between presented frames
    app = App(1000, None, WARMUP_FRAMES + SAMPLE_FRAMES + 1, parse_script(SCRIPT), view, cats=cats, cat_seed=16)
    renderer = renderer_string(app)
    render_size = list(app.screen.get_size())
    summary = app.run()
    # intervals between frames handed to the window (the app's own 60 Hz schedule; no vsync wait)
    frames = sorted(sample_intervals(app.frame_times))

    def pct(q):
        return 1000 * frames[min(int(q * len(frames)), len(frames) - 1)]
    return {"fps": len(frames) / sum(frames), "p50_ms": pct(0.50), "p95_ms": pct(0.95), "p99_ms": pct(0.99),
            "worst_ms": 1000 * frames[-1], "late_frames": sum(f > 1.5 / FPS for f in frames),
            "frames": len(frames), "rtf": summary["real_time_factor"], "renderer": renderer,
            "display_driver": app.display.driver, "render_size": render_size,
            "vsync": VSYNC, "window_mode": "fullscreen" if app.display.fullscreen else "windowed",
            "platform": platform.platform()}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--trials", type=int, default=3)
    p.add_argument("--cats", type=int, default=4, help="wandering cats during the run (default 4: the gate)")
    p.add_argument("--note", default="", help="background load or other conditions during the run")
    p.add_argument("--out", type=Path, default=OUT / "fps_protocol.json")
    p.add_argument("--gate", action="store_true", help="exit 1 unless every view meets the frame-rate gate")
    p.add_argument("--strict", action="store_true", help="release gate: refuse unless the commit is known, the tree clean, and nothing changes during the run")
    a = p.parse_args(argv)
    if a.trials < 3:
        p.error("at least 3 trials per view")
    if a.gate and a.cats != GATE["cats"]:
        p.error(f"the gate is measured with {GATE['cats']} cats")
    a.strict = a.strict or a.gate  # the frame-rate gate is a release gate: strict provenance
    from robot_env import provenance
    model = provenance.model_of(cats=a.cats, cat_seed=16)
    before = provenance.begin(__file__, sys.argv if argv is None else ["fps_protocol.py", *argv], a.strict, a.out,
                              model_sha256=model)
    result = {"protocol": 4, "cats": a.cats, "kernels": kernel_times(), "command": "python tools/fps_protocol.py " + " ".join(sys.argv[1:]),
              "time": time.strftime("%Y-%m-%d %H:%M:%S"), "warmup_frames": WARMUP_FRAMES,
              "sample_frames": SAMPLE_FRAMES, "script": SCRIPT, "window": list(WINDOW),
              "frame_budget_ms": 1000 / FPS, "renderer": None, "host": host_info(),
              "background_note": a.note or "unavailable", "views": {}}
    for view, name in enumerate(VIEWS):
        trials = [trial(view, a.cats) for _ in range(a.trials)]
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
    failed = [name for name, v in result["views"].items() if not (
        v["fps_min"] >= GATE["fps_min"] and v["p95_ms"] <= GATE["p95_ms_max"]
        and v["worst_ms"] <= GATE["worst_ms_max"] and v["rtf_min"] >= GATE["rtf_min"])]
    if a.gate:
        result["gate"] = {**GATE, "failed_views": failed, "result": "FAIL" if failed else "PASS"}
    after = provenance.end(before, __file__, sys.argv if argv is None else ["fps_protocol.py", *argv], a.strict,
                           a.out, model_sha256=provenance.model_of(cats=a.cats, cat_seed=16))
    result = {**after, **result}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    rev = result["build"]
    print(f"revision {rev['commit']} ({rev['working_tree']}), {a.cats} cats; kernels {result['kernels']}")
    if a.gate:
        print("frame-rate gate:", f"FAIL ({', '.join(failed)})" if failed else "PASS")
        return 1 if failed else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
