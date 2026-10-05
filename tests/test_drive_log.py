"""Drive log: versioned, lossless, failure-safe, and without effect on control."""

import json
import math

import numpy as np

from robot_env import config as C
from robot_env.drive_log import FORMAT, VERSION, DriveLog, read_log
from tests.helpers import empty_system


def _drive(s, seconds=2.0):
    for i in range(int(round(seconds / C.CONTROL_PERIOD))):
        s.drive(0.3, 0.5 if i > 40 else 0.0)
        s.advance(C.CONTROL_PERIOD)


def test_logging_does_not_change_the_run(tmp_path):
    runs = []
    for log in (None, tmp_path / "run.jsonl"):
        s = empty_system()
        if log:
            s.log = DriveLog(log, task_seed=7)
        s.reset(-1.5, 0.0, 0.0, (2.5, 2.5))
        _drive(s)
        runs.append((s.sim.data.qpos.copy(), s.applied, s.last_result))
        if s.log:
            s.log.close()
        s.close()
    assert np.array_equal(runs[0][0], runs[1][0]) and runs[0][1:] == runs[1][1:]


def test_log_round_trips_with_full_scans_and_telemetry(tmp_path):
    s = empty_system()
    s.log = DriveLog(tmp_path / "run.jsonl", task_seed=3, noise_seed=11, build="test-build")
    s.reset(-1.5, 0.0, 0.0, (2.5, 2.5))
    _drive(s, 1.0)
    last_scan = s._scan.copy()
    s.log.event("episode_timeout", s.time)
    s.log.close()
    recs = read_log(tmp_path / "run.jsonl")
    head, ticks, foot = recs[0], [r for r in recs if r["kind"] == "tick"], recs[-1]
    assert head["format"] == FORMAT and head["version"] == VERSION and head["build_label"] == "test-build"
    assert head["task_seed"] == 3 and head["noise_seed"] == 11 and len(head["config_sha256"]) == 64
    assert np.allclose(head["lidar_angles"], C.LIDAR_ANGLES)
    assert len(ticks) == round(1.0 / C.CONTROL_PERIOD) and foot["kind"] == "footer" and not foot["failed"]
    t = ticks[-1]
    assert np.array_equal(t["lidar"], last_scan)  # lossless
    assert t["applied"] == [s.applied.v, s.applied.omega] and t["reasons"] == list(s.last_result.reasons)
    assert t["command_expires"] == t["command_issued"] + C.COMMAND_LIFETIME
    assert "evaluation_only_truth" in t and "pose" not in t  # truth only under its own key
    assert [r["name"] for r in recs if r["kind"] == "event"] == ["reset", "episode_timeout"]
    assert [r["tick"] for r in ticks] == list(range(ticks[0]["tick"], ticks[0]["tick"] + len(ticks)))
    s.close()


def test_a_corrupted_scan_is_detected(tmp_path):
    import pytest
    s = empty_system()
    s.log = DriveLog(tmp_path / "run.jsonl")
    s.reset(-1.5, 0.0, 0.0, (2.5, 2.5))
    _drive(s, 0.1)
    s.log.close()
    lines = (tmp_path / "run.jsonl").read_text(encoding="utf-8").splitlines()
    rec = json.loads(lines[2])
    rec["lidar"]["crc32"] ^= 1
    lines[2] = json.dumps(rec)
    (tmp_path / "bad.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        read_log(tmp_path / "bad.jsonl")
    s.close()


class _FailingFile:
    def __init__(self, fail_on):
        self.fail_on, self.writes = fail_on, 0

    def write(self, text):
        self.writes += 1
        if self.fail_on == "write" and self.writes > 3:
            raise OSError("disk full")

    def close(self):
        if self.fail_on == "close":
            raise OSError("cannot close")


def test_write_and_close_failures_never_affect_control(tmp_path):
    reference = empty_system()
    reference.reset(-1.5, 0.0, 0.0, (2.5, 2.5))
    _drive(reference)
    for fail_on in ("write", "close"):
        s = empty_system()
        s.log = DriveLog(tmp_path / f"{fail_on}.jsonl")
        s.log._f = _FailingFile(fail_on)
        s.reset(-1.5, 0.0, 0.0, (2.5, 2.5))
        _drive(s)
        s.log.close()
        assert np.array_equal(s.sim.data.qpos, reference.sim.data.qpos)
        if fail_on == "write":
            assert s.log.failed and "disk full" in s.log.error
        else:
            assert s.log.failed and "cannot close" in s.log.error
        s.close()
    reference.close()


def test_open_failure_is_reported_not_raised(tmp_path):
    blocker = tmp_path / "blocker"
    blocker.write_text("x")
    log = DriveLog(blocker / "inside.jsonl")  # parent is a file: cannot create
    assert log.failed and log.error
    log.event("ignored", 0.0)
    log.close()


def test_invalid_numbers_are_logged_as_text(tmp_path):
    s = empty_system()
    s.log = DriveLog(tmp_path / "run.jsonl")
    s.reset(-1.5, 0.0, 0.0, (2.5, 2.5))
    s.drive(float("nan"), 0.0)
    s.advance(C.CONTROL_PERIOD)
    s.log.close()
    tick = [r for r in read_log(tmp_path / "run.jsonl") if r["kind"] == "tick"][-1]
    assert tick["requested"][0] == "nan" and "invalid_input" in tick["reasons"]
    assert not math.isnan(tick["applied"][0])
    s.close()


def _good_log(tmp_path, name="run.jsonl"):
    s = empty_system()
    with DriveLog(tmp_path / name) as log:
        s.log = log
        s.reset(-1.5, 0.0, 0.0, (2.5, 2.5))
        _drive(s, 0.2)
    s.log = None
    s.close()
    return (tmp_path / name).read_text(encoding="utf-8").splitlines()


def _write(tmp_path, lines, name="bad.jsonl"):
    (tmp_path / name).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return tmp_path / name


def test_truncated_failed_and_inconsistent_logs_are_rejected(tmp_path):
    import pytest
    from robot_env.drive_log import IncompleteLogError
    lines = _good_log(tmp_path)
    assert read_log(tmp_path / "run.jsonl")  # the complete log is valid
    bad = {
        "missing footer": lines[:-1],
        "missing header": lines[1:],
        "cut mid-line": lines[:-2] + [lines[-2][:40]],
        "dropped tick": lines[:3] + lines[4:],
        "duplicate tick": lines[:4] + [lines[3]] + lines[4:],
    }
    foot = json.loads(lines[-1])
    foot["ticks"] += 1
    bad["bad footer count"] = lines[:-1] + [json.dumps(foot)]
    foot = json.loads(lines[-1])
    foot["failed"], foot["error"] = True, "OSError: disk full"
    bad["failed while writing"] = lines[:-1] + [json.dumps(foot)]
    for name, content in bad.items():
        with pytest.raises(IncompleteLogError):
            read_log(_write(tmp_path, content))
    # a truncated log can still be read for diagnosis, explicitly
    assert read_log(_write(tmp_path, lines[:-1]), allow_incomplete=True)[-1]["kind"] != "footer"


def test_closing_the_system_finalizes_its_log(tmp_path):
    s = empty_system()
    s.log = DriveLog(tmp_path / "run.jsonl")
    s.reset(-1.5, 0.0, 0.0, (2.5, 2.5))
    _drive(s, 0.1)
    s.close()
    assert read_log(tmp_path / "run.jsonl")[-1]["kind"] == "footer"
