"""What ran is recorded: build identity, the exact model input, dependencies; release gates refuse
an unknown or modified build or a change during the run (robot_env/provenance.py)."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from robot_env import provenance as P
from robot_env.drive_log import DriveLog, IncompleteLogError, provenance_of, read_log, validate_log
from robot_env.sim import RobotSim
from robot_env.system import RobotSystem


@pytest.fixture(autouse=True)
def fresh():
    P.reset_cache()
    yield
    P.reset_cache()


def test_identical_models_hash_alike_and_any_input_change_shows():
    a, b = RobotSim().model_sha256, RobotSim().model_sha256
    assert a == b and len(a) == 64
    assert RobotSim(include_obstacles=False).model_sha256 != a  # obstacle mode
    assert RobotSim(extra_world_xml='<geom name="x" type="box" size="0.1 0.1 0.1" pos="9 9 1"/>').model_sha256 != a
    one, two = RobotSystem(cats=1, cat_seed=0), RobotSystem(cats=2, cat_seed=0)
    assert len({a, one.sim.model_sha256, two.sim.model_sha256}) == 3  # the cat count
    one.close()
    two.close()


def test_the_model_hash_covers_asset_bytes_names_and_framing():
    xml = "<mujoco/>"
    base = P.model_sha256(xml, {"t.png": b"\x00\x01"})
    assert P.model_sha256(xml, {"t.png": b"\x00\x02"}) != base  # one asset byte
    assert P.model_sha256(xml, {"u.png": b"\x00\x01"}) != base  # an asset name
    assert P.model_sha256(xml, {"a": b"bc"}) != P.model_sha256(xml, {"ab": b"c"})  # framed, not concatenated
    assert P.model_sha256(xml, {"t.png": b"\x00\x01", "u.png": b""}) != base  # an extra asset
    assert P.model_sha256(xml, {"t.png": b"\x00\x01"}) == base  # deterministic


def test_without_cats_the_cat_assets_are_not_part_of_the_model(monkeypatch):
    """An unused file (the cat coats when no cat is loaded) cannot change the model hash."""
    from robot_env import cats as K
    plain = RobotSystem(cats=0).sim.model_sha256
    monkeypatch.setattr(K, "COATS", tuple(reversed(K.COATS)))  # a different cat set-up, not loaded
    assert RobotSystem(cats=0).sim.model_sha256 == plain


@pytest.mark.parametrize("rev,status,commit,tree", [
    (None, None, None, "unknown"),  # no git at all
    ("not-a-commit\n", "", None, "unknown"),  # malformed output
    ("a" * 40 + "\n", "", "a" * 40, "clean"),
    ("a" * 40 + "\n", " M robot_env/cats.py\n", "a" * 40, "modified"),
    ("a" * 40 + "\n", "?? tools/new_tool.py\n", "a" * 40, "modified"),  # an untracked, not ignored file
    ("a" * 40 + "\n", None, "a" * 40, "unknown"),  # git status failed
])
def test_build_identity_without_or_with_git(monkeypatch, rev, status, commit, tree):
    monkeypatch.setattr(P, "_git", lambda *a: rev if a[0] == "rev-parse" else status)
    b = P.build_id(refresh=True)
    assert b["commit"] == commit and b["working_tree"] == tree
    assert b["source_sha256"] is not None and len(b["source_sha256"]) == 64  # git-independent


def test_git_failures_never_raise(monkeypatch):
    def boom(*a, **k):
        raise OSError("no git")
    monkeypatch.setattr(P.subprocess, "run", boom)
    assert P._git("status") is None
    monkeypatch.setattr(P.subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(subprocess.TimeoutExpired("git", 10)))
    assert P._git("status") is None


def test_source_hash_changes_with_a_source_file_and_ignores_caches(tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "a.py").write_text("x = 1\n")
    first = P.source_sha256(tmp_path)
    (tmp_path / "pkg" / "__pycache__").mkdir()
    (tmp_path / "pkg" / "__pycache__" / "a.cpython-311.pyc").write_bytes(b"cache")
    (tmp_path / "pkg" / "k.nbi").write_bytes(b"numba index")
    assert P.source_sha256(tmp_path) == first  # caches do not count
    (tmp_path / "pkg" / "a.py").write_text("x = 2\n")
    assert P.source_sha256(tmp_path) != first


def test_strict_gate_refuses_unknown_or_modified_builds_and_changes_during_the_run(monkeypatch, tmp_path):
    clean = {"build": {"commit": "a" * 40, "working_tree": "clean", "source_sha256": "1" * 64},
             "tool_sha256": "2" * 64, "model_sha256": "3" * 64}
    assert P.check_strict(clean) is None and P.check_strict(clean, dict(clean)) is None
    assert "unknown" in P.check_strict({**clean, "build": {**clean["build"], "commit": None}})
    assert "modified" in P.check_strict({**clean, "build": {**clean["build"], "working_tree": "modified"}})
    for key, value in (("tool_sha256", "9" * 64), ("model_sha256", "9" * 64)):
        assert "changed" in P.check_strict(clean, {**clean, key: value})
    assert "changed" in P.check_strict(clean, {**clean, "build": {**clean["build"], "source_sha256": "9" * 64}})
    # begin(): a modified tree refuses the run, writes a refusal record, and exits with status 2
    monkeypatch.setattr(P, "_git", lambda *a: "a" * 40 + "\n" if a[0] == "rev-parse" else " M x.py\n")
    out = tmp_path / "result.json"
    with pytest.raises(SystemExit) as e:
        P.begin(__file__, ["tool", "--strict"], True, out, model_sha256="3" * 64)
    assert e.value.code == 2
    record = json.loads(out.read_text())
    assert record["result"] == "REFUSED" and "modified" in record["refusal"] and record["argv"] == ["tool", "--strict"]
    assert P.begin(__file__, ["tool"], False, None)["build"]["working_tree"] == "modified"  # recorded, not refused


def test_a_change_during_a_strict_run_is_refused_at_the_end(monkeypatch, tmp_path):
    monkeypatch.setattr(P, "_git", lambda *a: "a" * 40 + "\n" if a[0] == "rev-parse" else "")
    before = P.begin(__file__, ["tool"], True, None, model_sha256="3" * 64)
    with pytest.raises(SystemExit):
        P.end(before, __file__, ["tool"], True, tmp_path / "r.json", model_sha256="4" * 64)  # the model changed
    assert json.loads((tmp_path / "r.json").read_text())["result"] == "REFUSED"


def test_drive_log_v2_records_provenance_and_v1_logs_still_read(tmp_path):
    s = RobotSystem(cats=1, cat_seed=3)
    s.log = DriveLog(tmp_path / "run.jsonl", build="label")
    s.reset(-4.0, -2.5, 0.0, (4.0, 2.5))
    s.advance(0.1)
    s.close()
    recs = read_log(tmp_path / "run.jsonl")
    head = recs[0]
    assert head["version"] == 2 and head["build_label"] == "label"
    assert head["model_sha256"] == RobotSystem(cats=1, cat_seed=3).sim.model_sha256
    assert set(head["build"]) == {"commit", "working_tree", "source_sha256"}
    assert head["deps"]["kernels_compiled"] in (True, False) and head["deps"]["mujoco"]
    # the same log as version 1 (no provenance): readable, provenance unknown
    raw = [json.loads(line) for line in (tmp_path / "run.jsonl").read_text().splitlines()]
    v1 = dict(raw[0])
    for key in ("build", "build_label", "model_sha256", "deps"):
        v1.pop(key)
    v1["version"], v1["build"] = 1, "old"
    (tmp_path / "v1.jsonl").write_text("\n".join(json.dumps(r) for r in [v1, *raw[1:]]) + "\n")
    old = read_log(tmp_path / "v1.jsonl")
    p = provenance_of(old[0])
    assert p["build"] == {"commit": None, "working_tree": "unknown", "source_sha256": None} and p["build_label"] == "old"
    # a v2 header with a malformed hash is refused
    bad = dict(raw[0], model_sha256="xyz")
    with pytest.raises(IncompleteLogError):
        validate_log([bad, *[r for r in raw[1:]]])


def test_collecting_provenance_imports_no_kernels():
    code = ("import sys; from robot_env import provenance as P; P.build_id(); d = P.deps(); "
            "print('robot_env.kernels' in sys.modules, 'numba' in sys.modules, d['kernels_compiled'])")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=P.ROOT, timeout=120)
    assert out.stdout.split() == ["False", "False", "None"]
