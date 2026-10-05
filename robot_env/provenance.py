"""What exactly ran: the code, the model, and the libraries, for drive logs and evaluation results.

- build_id(): the git commit (40 hex digits, or None), whether the working tree differs from it
  (clean, modified, or unknown), and a git-independent SHA-256 of the robot_env source files.
- model_sha256(xml, assets): SHA-256 of the exact model input given to MuJoCo: the final XML
  string and every in-memory asset beside it (names and bytes). RobotSim records it.
- deps(): versions of Python and the packages that shape a run, and whether the cats' kernels
  are compiled.
- file_sha256(path): one file (an evaluation tool's own source).
- artifact(tool, argv, ...) and check_strict(before, after): the provenance block of an
  evaluation result, and the release-gate rule: a clean, known commit, unchanged during the run.

Nothing here raises for a missing git, a timeout, odd output, or an unreadable file: those give
None or "unknown". Nothing here imports or compiles the kernels (it only reports them if loaded).
"""

from __future__ import annotations

import hashlib
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE_DIR = Path(__file__).resolve().parent
SCHEMA = 1  # the provenance block's own version
_EXCLUDED_PARTS = {"__pycache__"}
_EXCLUDED_SUFFIXES = {".pyc", ".pyo", ".nbi", ".nbc"}
_PACKAGES = ("mujoco", "numpy", "numba", "llvmlite", "gymnasium", "pygame")
_cache: dict = {}


def reset_cache() -> None:
    """Forget the per-process build identity (tests)."""
    _cache.clear()


def _frame(h, data: bytes) -> None:
    h.update(len(data).to_bytes(8, "little"))
    h.update(data)


def model_sha256(xml: str, assets: dict) -> str:
    """SHA-256 of the exact model input: the XML text and each asset (sorted by name), every
    piece length-framed so no two different inputs hash alike by concatenation."""
    h = hashlib.sha256()
    _frame(h, xml.encode("utf-8"))
    for name in sorted(assets):
        _frame(h, name.encode("utf-8"))
        _frame(h, bytes(assets[name]))
    return h.hexdigest()


def source_sha256(root: Path = SOURCE_DIR) -> str | None:
    """SHA-256 over the source files under root (sorted relative POSIX paths and bytes, framed),
    without caches or compiled files. None if a file cannot be read."""
    try:
        files = sorted(p for p in root.rglob("*") if p.is_file() and not (_EXCLUDED_PARTS & set(p.parts))
                       and p.suffix not in _EXCLUDED_SUFFIXES)
        h = hashlib.sha256()
        for p in files:
            _frame(h, p.relative_to(root).as_posix().encode("utf-8"))
            _frame(h, p.read_bytes())
        return h.hexdigest()
    except OSError:
        return None


def file_sha256(path: str | Path) -> str | None:
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return None


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError, ValueError):
        return None
    return out.stdout if out.returncode == 0 else None


def build_id(refresh: bool = False) -> dict:
    """{"commit", "working_tree", "source_sha256"}; cached for the process unless refresh."""
    if refresh or "build" not in _cache:
        commit = _git("rev-parse", "HEAD")
        commit = commit.strip() if commit else None
        if commit is not None and (len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit)):
            commit = None
        status = _git("status", "--porcelain", "--untracked-files=normal") if commit else None
        tree = "unknown" if status is None else ("modified" if status.strip() else "clean")
        _cache["build"] = {"commit": commit, "working_tree": tree, "source_sha256": source_sha256()}
    return dict(_cache["build"])


def _version(name: str) -> str | None:
    try:
        from importlib.metadata import PackageNotFoundError, version
        try:
            return version(name)
        except PackageNotFoundError:
            return None
    except Exception:
        return None


def deps() -> dict:
    """Package versions (None if absent) and the kernels' mode in the same snapshot: compiled,
    interpreted, or None when the kernels were not loaded in this process."""
    kernels = sys.modules.get("robot_env.kernels")
    return {"python": platform.python_version(), **{name: _version(name) for name in _PACKAGES},
            "kernels_compiled": None if kernels is None else bool(getattr(kernels, "COMPILED", False))}


def artifact(tool: str | Path, argv: list[str], **fields) -> dict:
    """The provenance block of an evaluation result (schema, build, the tool's own hash, the
    exact command line, dependencies, and whatever the caller adds: seeds, configuration)."""
    return {"provenance_schema": SCHEMA, "build": build_id(refresh=True), "tool": Path(tool).name,
            "tool_sha256": file_sha256(tool), "argv": list(argv), "deps": deps(), **fields}


def check_strict(before: dict, after: dict | None = None) -> str | None:
    """None if a release-gate run may stand: a known commit, a clean tree, and (given the end
    snapshot) nothing changed during the run; otherwise the reason it may not."""
    b = before["build"]
    if b.get("commit") is None:
        return "the commit is unknown"
    if b.get("working_tree") != "clean":
        return f"the working tree is {b.get('working_tree')}"
    if after is not None:
        a = after["build"]
        if a != b or after.get("tool_sha256") != before.get("tool_sha256") or \
                after.get("model_sha256") != before.get("model_sha256"):
            return "the code or the model changed during the run"
    return None


def model_of(cats: int = 0, cat_seed: int = 0) -> str | None:
    """model_sha256 of the world a tool runs (built once here; None if it cannot be built)."""
    try:
        from .system import RobotSystem
        s = RobotSystem(cats=cats, cat_seed=cat_seed)
        try:
            return s.sim.model_sha256
        finally:
            s.close()
    except Exception:
        return None


def begin(tool, argv, strict: bool, out=None, **fields) -> dict:
    """The provenance snapshot at the start of an evaluation. With strict (a release gate), an
    unknown commit or a modified tree refuses the run: a refusal record is written to `out` if
    given, and the process exits with status 2."""
    snap = artifact(tool, argv, strict=strict, **fields)
    if strict:
        reason = check_strict(snap)
        if reason:
            refuse(snap, reason, out)
    return snap


def end(before: dict, tool, argv, strict: bool, out=None, **fields) -> dict:
    """The snapshot at the end; with strict, the run is refused if anything changed meanwhile."""
    after = artifact(tool, argv, strict=strict, **fields)
    if strict:
        reason = check_strict(before, after)
        if reason:
            refuse(after, reason, out)
    return after


def refuse(snap: dict, reason: str, out=None) -> None:
    import json
    record = {**snap, "result": "REFUSED", "refusal": reason}
    if out is not None:
        try:
            Path(out).parent.mkdir(parents=True, exist_ok=True)
            Path(out).write_text(json.dumps(record, indent=1), encoding="utf-8")
        except OSError:
            pass
    print(f"REFUSED (release gate): {reason}", flush=True)
    raise SystemExit(2)
