"""Drive log: one versioned JSON-lines file per run, for replay and safety diagnosis.

Opt-in. Attach with `RobotSystem.log = DriveLog(path, ...)`; the system calls `tick()` at the end
of every 50 Hz control tick, after the command for that tick is decided, so logging can never
change what the robot does. Every write failure is caught: the log marks itself failed and stops
writing, and the run continues unchanged.

Record kinds: "header" (first line), "tick" (each control tick), "event" (episode transitions and
anything the caller reports), "footer" (last line, with counts and the failure state).
The full lidar scan is stored losslessly (float64 little-endian, base64) with a CRC-32 checksum.
Ground truth appears only under the key "evaluation_only_truth" and must never reach a driver.

Finalize every log: use `with DriveLog(...) as log:` or call `close()` in a `finally` block
(RobotSystem.close() also closes an attached log). `read_log()` rejects a log that is incomplete
(no footer, for example after a crash), failed, or inconsistent, with IncompleteLogError.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import platform
import zlib
from pathlib import Path

import numpy as np

from . import config as C
from . import provenance

FORMAT = "robot-drive-log"
VERSION = 3  # 2: header provenance: build (commit, working tree, source hash), build_label, model_sha256, deps;
            # 3: each new forward depth frame in the tick after it is taken (depth in mm, validity, time, IMU up)
READABLE = (1, 2, 3)  # version 1 logs (no provenance) are still read; their provenance is reported unknown
UNITS = {"t": "s", "v": "m/s", "omega": "rad/s", "lidar": "m", "lidar_angles": "rad",
         "pose": "m, m, rad (yaw)", "velocity": "m/s, rad/s"}


def config_hash() -> str:
    """SHA-256 of every upper-case constant in robot_env.config (the profile's numbers)."""
    values = {k: getattr(C, k) for k in sorted(dir(C)) if k.isupper()}
    text = json.dumps(values, sort_keys=True, default=repr)
    return hashlib.sha256(text.encode()).hexdigest()


def pack_array(a: np.ndarray, dtype: str) -> dict:
    raw = np.ascontiguousarray(a, dtype=dtype).tobytes()
    return {"dtype": dtype, "n": int(a.size), "b64": base64.b64encode(raw).decode(), "crc32": zlib.crc32(raw)}


def unpack_array(d: dict) -> np.ndarray:
    raw = base64.b64decode(d["b64"])
    if zlib.crc32(raw) != d["crc32"]:
        raise ValueError("drive log array checksum mismatch")
    return np.frombuffer(raw, dtype=d["dtype"]).copy()


def _cmd(c) -> list[float] | None:
    return None if c is None else [float(c.v), float(c.omega)]


def _finite(x):
    return x if x is None or math.isfinite(x) else str(x)


class IncompleteLogError(ValueError):
    """The log is truncated, failed while writing, or internally inconsistent."""


class DriveLog:
    def __init__(self, path: str | Path, *, profile: str = "ideal", task_seed: int | None = None,
                 noise_seed: int | None = None, build: str = "", note: str = "",
                 cats: int | None = None, cat_seed: int | None = None):
        self.path = Path(path)
        self.failed = False
        self.error: str | None = None
        self.ticks = 0
        self.events = 0
        self._f = None
        self._header_written = False
        self._claimed = {"cats": cats, "cat_seed": cat_seed}  # what the caller says (checked at bind)
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._f = open(self.path, "w", encoding="utf-8", newline="\n")
        except OSError as e:
            self._fail(e)
            return
        self._header = {
            "kind": "header", "format": FORMAT, "version": VERSION, "units": UNITS,
            "profile": profile, "config_sha256": config_hash(),
            "build": provenance.build_id(), "build_label": build or "",
            "model_sha256": None, "deps": None,  # filled from the system at bind
            "task_seed": task_seed, "noise_seed": noise_seed,
            "physics_dt": C.PHYSICS_DT, "control_period": C.CONTROL_PERIOD, "decision_period": C.DECISION_PERIOD,
            "command_lifetime": C.COMMAND_LIFETIME, "scan_max_age": C.SCAN_MAX_AGE,
            "lidar_angles": pack_array(np.array(C.LIDAR_ANGLES), "<f8"),
            "host": {"python": platform.python_version(), "platform": platform.platform(),
                     "machine": platform.machine() or "unavailable"},
            "note": note,
            "cats": cats or 0, "cat_seed": cat_seed,
        }

    def _fail(self, e: Exception) -> None:
        self.failed = True
        self.error = f"{type(e).__name__}: {e}"
        try:
            if self._f is not None:
                self._f.close()
        except OSError:
            pass
        self._f = None

    @property
    def started(self) -> bool:
        """True once any record is written (a log holds one episode: reset must not reuse it)."""
        return self._header_written

    def bind(self, system) -> None:
        """Fill the header from the system it logs (cat count and seed). A value the caller gave
        that disagrees with the system makes the log fail (it would describe a different run)."""
        herd = getattr(system, "cats", None)
        actual = {"cats": herd.n if herd is not None else 0, "cat_seed": herd.seed if herd is not None else None}
        for key, claimed in self._claimed.items():
            if claimed is not None and claimed != actual[key]:
                self._fail(ValueError(f"log says {key}={claimed} but the system has {actual[key]}"))
                return
        try:
            actual["model_sha256"] = system.sim.model_sha256
        except Exception:  # provenance never stops a run
            actual["model_sha256"] = None
        actual["deps"] = provenance.deps()
        if self._header_written:
            if any(self._header.get(k) != actual[k] for k in ("cats", "cat_seed", "model_sha256")):
                self._fail(ValueError("log header written for a different system"))
            return
        self._header.update(actual)

    def _write(self, record: dict) -> None:
        if self.failed or self._f is None:
            return
        if not self._header_written:
            self._header_written = True
            self._write(self._header)
            if self.failed:
                return
        try:
            self._f.write(json.dumps(record, separators=(",", ":")) + "\n")
        except (OSError, ValueError) as e:
            self._fail(e)

    def tick(self, system) -> None:
        """One record per control tick (called by RobotSystem; never raises)."""
        if self.failed:
            return
        if not self._header_written:
            self.bind(system)
        try:
            res = system.last_result
            issued = system._issued_at
            v_est, w_est = system.sim.velocity_estimate()
            x, y, yaw = system.sim.true_pose()
            tv, tw = system.sim.true_velocity()
            self._write({
                "kind": "tick", "t": system.time, "tick": system._step // system._ctrl_every,
                "scan_time": system._scan_time,
                "command_issued": issued,
                "command_expires": None if issued is None else issued + C.COMMAND_LIFETIME,
                "requested": None if system._requested is None else
                [_finite(system._requested.v), _finite(system._requested.omega)],
                "approved": _cmd(res.command), "applied": _cmd(system.applied),
                "reasons": list(res.reasons), "contact": bool(system._contact),
                "collisions": system.collisions, "intervention_events": system.intervention_events,
                "encoder": [v_est, w_est],
                "lidar": pack_array(system._scan, "<f8"), "lidar_valid": pack_array(system._scan_valid, "u1"),
                **self._depth_frame(system),
                "flags": {k: bool(getattr(system.flags, k)) for k in
                          ("emergency_brake", "focus_lost", "episode_over", "manual_mode", "manual_input_held")},
                "evaluation_only_truth": {"pose": [x, y, yaw], "velocity": [tv, tw],
                                          **({"cats": system.cats.truth(),
                                              "cat_contacts": system.cat_contacts}
                                             if getattr(system, "cats", None) is not None else {})},
            })
            self.ticks += 1
        except Exception as e:  # a logging bug must never stop the robot's control
            self._fail(e)

    def _depth_frame(self, system) -> dict:
        """The depth frame, once: in the first tick after it was taken (millimetres, uint16)."""
        t = getattr(system, "_depth_time", None)
        if t is None or t == getattr(self, "_logged_depth", None):
            return {}
        self._logged_depth = t
        mm = np.clip(np.round(np.asarray(system._depth) * 1000.0), 0, 65535)
        return {"depth": pack_array(mm, "<u2"), "depth_valid": pack_array(system._depth_valid, "u1"),
                "depth_time": t, "depth_up": [float(v) for v in system._depth_up]}

    def event(self, name: str, t: float, **data) -> None:
        self._write({"kind": "event", "t": t, "name": name, **data})
        if not self.failed:
            self.events += 1

    def __enter__(self) -> "DriveLog":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        if self._f is None:
            return
        self._write({"kind": "footer", "ticks": self.ticks, "events": self.events,
                     "failed": self.failed, "error": self.error})
        try:
            self._f.close()
        except OSError as e:
            self._fail(e)
        self._f = None


def read_log(path: str | Path, allow_incomplete: bool = False) -> list[dict]:
    """All records, with the lidar arrays decoded (checksums verified) and the log validated:
    format and version, exactly one header first and one footer last, footer counts equal to
    the records, contiguous tick numbers, non-decreasing time, and scan lengths equal to the
    header's lidar angles. Raises IncompleteLogError otherwise (`allow_incomplete=True` returns a
    truncated log's records for diagnosis, still checking everything that is present)."""
    out = []
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            try:
                rec = json.loads(line, parse_constant=_reject_constant)
            except _NonFinite as e:
                raise IncompleteLogError(f"line {n}: non-finite number {e}") from e
            except json.JSONDecodeError as e:
                raise IncompleteLogError(f"line {n}: not JSON (truncated write?)") from e
            if not isinstance(rec, dict):
                raise IncompleteLogError(f"line {n}: not a record")
            for key in ("lidar", "lidar_valid", "lidar_angles", "depth", "depth_valid"):
                if isinstance(rec.get(key), dict) and "b64" in rec[key]:
                    rec[key] = unpack_array(rec[key])
            out.append(rec)
    validate_log(out, allow_incomplete)
    return out


class _NonFinite(ValueError):
    pass


def _reject_constant(name: str):
    raise _NonFinite(name)  # NaN, Infinity, -Infinity are not valid log numbers


def _check_time(rec: dict, key: str, required: bool) -> None:
    """A timestamp must be a real finite number (not bool, not text); optional ones may be None."""
    if key not in rec:
        if required:
            raise IncompleteLogError(f"{rec.get('kind')} record without {key!r}")
        return
    v = rec[key]
    if v is None and not required:
        return
    try:
        ok = not isinstance(v, bool) and isinstance(v, (int, float)) and math.isfinite(v)
    except OverflowError:  # an integer too large for a float (e.g. 401 digits)
        ok = False
    if not ok:
        raise IncompleteLogError(f"{rec.get('kind')} record: {key}={str(v)[:40]!r} is not a finite number")


def _hex(v, n: int) -> bool:
    return isinstance(v, str) and len(v) == n and all(c in "0123456789abcdef" for c in v)


def _check_provenance(head: dict) -> None:
    """A version 2 header's provenance: types and hash formats (None where unknown)."""
    b = head.get("build")
    if not isinstance(b, dict) or set(b) != {"commit", "working_tree", "source_sha256"}:
        raise IncompleteLogError("header build is not {commit, working_tree, source_sha256}")
    if b["commit"] is not None and not _hex(b["commit"], 40):
        raise IncompleteLogError("header build commit is not 40 hex digits")
    if b["working_tree"] not in ("clean", "modified", "unknown"):
        raise IncompleteLogError("header build working_tree is not clean, modified, or unknown")
    for key, value in (("source_sha256", b["source_sha256"]), ("model_sha256", head.get("model_sha256"))):
        if value is not None and not _hex(value, 64):
            raise IncompleteLogError(f"header {key} is not 64 hex digits")
    if not isinstance(head.get("build_label"), str):
        raise IncompleteLogError("header build_label is not text")
    d = head.get("deps")
    if d is not None:
        keys = {"python", *provenance._PACKAGES, "kernels_compiled"}
        if not isinstance(d, dict) or set(d) != keys:
            raise IncompleteLogError("header deps is not a dependency snapshot")
        if any(d[k] is not None and not isinstance(d[k], str) for k in keys - {"kernels_compiled"}):
            raise IncompleteLogError("header deps: a version is not text")
        if d["kernels_compiled"] is not None and not isinstance(d["kernels_compiled"], bool):  # not 0, 1, 1.0
            raise IncompleteLogError("header deps: kernels_compiled is not true, false, or null")


def provenance_of(head: dict) -> dict:
    """A log header's provenance; for a version 1 log (written before it was recorded), unknown."""
    if head.get("version", 1) < 2:
        return {"build": {"commit": None, "working_tree": "unknown", "source_sha256": None},
                "build_label": head.get("build", ""), "model_sha256": None, "deps": None}
    return {k: head.get(k) for k in ("build", "build_label", "model_sha256", "deps")}


def validate_log(records: list[dict], allow_incomplete: bool = False) -> None:
    if not records or records[0].get("kind") != "header":
        raise IncompleteLogError("missing header")
    head = records[0]
    if head.get("format") != FORMAT or head.get("version") not in READABLE:
        raise IncompleteLogError(f"unsupported format {head.get('format')} v{head.get('version')}")
    if head["version"] >= 2:
        _check_provenance(head)
    kinds = [r.get("kind") for r in records]
    if kinds.count("header") != 1:
        raise IncompleteLogError("more than one header")
    has_footer = kinds[-1] == "footer"
    if kinds.count("footer") > 1 or ("footer" in kinds and not has_footer):
        raise IncompleteLogError("footer is not the last record")
    if not has_footer and not allow_incomplete:
        raise IncompleteLogError("missing footer: the log was not closed (crash or interrupted run)")
    ticks = [r for r in records if r.get("kind") == "tick"]
    events = [r for r in records if r.get("kind") == "event"]
    if has_footer:
        foot = records[-1]
        if foot.get("failed"):
            raise IncompleteLogError(f"the log failed while writing: {foot.get('error')}")
        if foot.get("ticks") != len(ticks) or foot.get("events") != len(events):
            raise IncompleteLogError(f"footer counts {foot.get('ticks')} ticks / {foot.get('events')} events, "
                                     f"file has {len(ticks)} / {len(events)}")
    for r in records[1:]:
        if r.get("kind") in ("tick", "event"):
            _check_time(r, "t", required=True)
        if r.get("kind") == "tick":
            _check_time(r, "scan_time", required=True)
            _check_time(r, "command_issued", required=False)
            _check_time(r, "command_expires", required=False)
            if not isinstance(r.get("tick"), int) or isinstance(r.get("tick"), bool):
                raise IncompleteLogError(f"tick record without an integer tick number: {r.get('tick')!r}")
    n_rays = len(head["lidar_angles"])
    for prev, cur in zip(ticks, ticks[1:]):
        if cur["tick"] != prev["tick"] + 1:
            raise IncompleteLogError(f"tick {cur['tick']} follows tick {prev['tick']} (missing or out of order)")
    times = [r["t"] for r in records[1:] if "t" in r]
    if any(b < a - 1e-12 for a, b in zip(times, times[1:])):
        raise IncompleteLogError("time goes backwards")
    for t in ticks:
        if len(t["lidar"]) != n_rays or len(t["lidar_valid"]) != n_rays:
            raise IncompleteLogError(f"tick {t['tick']}: scan has {len(t['lidar'])} rays, header says {n_rays}")
