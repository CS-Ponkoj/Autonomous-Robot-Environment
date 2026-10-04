"""Run the focused cat checks on all three kernel paths and compare (development gate):
compiled (numba), interpreted (NUMBA_DISABLE_JIT=1: the same kernel source as plain Python), and
without numba at all (its import blocked: the fallback decorator).

On each path: the cat and kernel tests (tests/test_cats.py, tests/test_kernels.py, without the
slow test) and tools/gait_check.py. Then the slow test itself, which runs one scripted
simulation on all three paths and compares their digests (every collider, the robot's state,
and every cat's state, commands, guard decisions, route, and rejected commands, at every control
tick). Gate: every run passes. The interpreted paths are slow (about 20 times the compiled one);
the three run in parallel.

    .venv\\Scripts\\python tools\\kernel_paths_check.py
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PATHS = {
    "compiled": {},
    "interpreted (NUMBA_DISABLE_JIT=1)": {"NUMBA_DISABLE_JIT": "1"},
    "without numba": {"PYTHONPATH": str(ROOT / "tests" / "_no_numba")},
}
COMMANDS = (
    [sys.executable, "-m", "pytest", "tests/test_kernels.py", "tests/test_cats.py", "-m", "not slow", "-q", "-p",
     "no:cacheprovider"],
    [sys.executable, "tools/gait_check.py"],
)


def run(name: str, env: dict) -> tuple[str, list[tuple[str, int, str, float]]]:
    out = []
    for cmd in COMMANDS:
        t = time.perf_counter()
        r = subprocess.run(cmd, cwd=ROOT, env={**os.environ, **env}, capture_output=True, text=True)
        tail = (r.stdout.strip().splitlines() or [""])[-1]
        out.append((" ".join(cmd[1:4]), r.returncode, tail, time.perf_counter() - t))
    probe = subprocess.run([sys.executable, "-c", "from robot_env import kernels; print(kernels.COMPILED)"],
                           cwd=ROOT, env={**os.environ, **env}, capture_output=True, text=True)
    out.append(("kernels.COMPILED", 0, probe.stdout.strip(), 0.0))
    return name, out


def main() -> int:
    with ThreadPoolExecutor(len(PATHS)) as pool:
        results = list(pool.map(lambda kv: run(*kv), PATHS.items()))
    t = time.perf_counter()
    trace = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                            "tests/test_kernels.py::test_compiled_interpreted_and_numba_free_runs_are_identical"],
                           cwd=ROOT, capture_output=True, text=True)
    results.append(("all three paths, one trace", [("cross-path trace digest", trace.returncode,
                    (trace.stdout.strip().splitlines() or [""])[-1], time.perf_counter() - t)]))
    ok = True
    for name, runs in results:
        print(name)
        for what, code, tail, secs in runs:
            print(f"   {what:40s} exit {code}  {secs:7.0f} s  {tail}")
            ok &= code == 0
        if name in PATHS:
            ok &= runs[-1][2] == ("True" if name == "compiled" else "False")
    print("kernel paths:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
