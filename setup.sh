#!/usr/bin/env bash
# Creates the project Python environment (.venv) and installs the pinned packages (macOS, Linux).
# Usage, in this folder:  ./setup.sh      (the Windows equivalent is setup.ps1)
set -euo pipefail
cd "$(dirname "$0")"

PY="${PYTHON:-}"
if [ -z "$PY" ]; then
    for candidate in python3.11 python3; do
        if command -v "$candidate" >/dev/null 2>&1 && \
           "$candidate" -c 'import sys; sys.exit(0 if sys.version_info[:2] == (3, 11) else 1)'; then
            PY="$candidate"
            break
        fi
    done
fi
if [ -z "$PY" ]; then
    echo "Python 3.11 is required and was not found (set PYTHON=/path/to/python3.11 to choose one)." >&2
    exit 1
fi
"$PY" --version

if [ ! -x .venv/bin/python ]; then
    "$PY" -m venv .venv
fi
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt -c requirements.lock
.venv/bin/python -c "import mujoco, gymnasium, pygame, numba; from robot_env import kernels; t = kernels.warm(); print('Setup OK: MuJoCo', mujoco.__version__, '/ Gymnasium', gymnasium.__version__, '/ Numba', numba.__version__, '(cat kernels compiled in %.1f s)' % t)"
