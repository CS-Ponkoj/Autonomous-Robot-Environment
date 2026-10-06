# Creates the project Python environment (.venv) and installs the pinned packages.
# Usage (PowerShell, in this folder):  .\setup.ps1
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
    throw "The Python launcher 'py' was not found. Install Python 3.11 from python.org."
}
& py -3.11 --version
if ($LASTEXITCODE -ne 0) { throw "Python 3.11 is required (py -3.11 failed)." }

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    & py -3.11 -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw "Creating .venv failed." }
}
& .venv\Scripts\python.exe -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "Upgrading pip failed." }
& .venv\Scripts\python.exe -m pip install -r requirements.txt -c requirements.lock
if ($LASTEXITCODE -ne 0) { throw "Installing requirements failed." }
& .venv\Scripts\python.exe -c "import mujoco, gymnasium, pygame, numba; from robot_env import kernels; t = kernels.warm(); print('Setup OK: MuJoCo', mujoco.__version__, '/ Gymnasium', gymnasium.__version__, '/ Numba', numba.__version__, '(cat kernels compiled in %.1f s)' % t)"
if ($LASTEXITCODE -ne 0) { throw "Package check failed." }
