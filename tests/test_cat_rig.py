"""The realistic cat rig: conversion integrity, licence manifest, bind-pose reproduction in
MuJoCo's skinning, forward kinematics, and real cat proportions."""

import hashlib
import json

import mujoco
import numpy as np
import pytest

from robot_env.cat_rig import CAT_DIR, Skeleton, load_rig, model_parts

TEXTURES = {"anisotropic1": "cat_anisotropic1.png", "anisotropic2": "cat_anisotropic2.png",
            "anisotropic3": "cat_anisotropic3.png"}


def _model(prefix="cat0_"):
    a, b, s, files = model_parts(prefix, TEXTURES)
    xml = (f"<mujoco><asset>{a}</asset><worldbody><light pos='0 0 2'/><geom type='plane' size='1 1 .1'/>{b}"
           f"</worldbody><deformable>{s}</deformable></mujoco>")
    m = mujoco.MjModel.from_xml_string(xml, files)
    return m, mujoco.MjData(m)


def _skin_vertices(m, d):
    scn = mujoco.MjvScene(m, 100)
    mujoco.mjv_updateScene(m, d, mujoco.MjvOption(), None, mujoco.MjvCamera(), mujoco.mjtCatBit.mjCAT_ALL, scn)
    return np.array(scn.skinvert).reshape(-1, 3)


def test_rig_matches_the_source_model():
    rig = load_rig()
    assert len(rig.joint_names) == 34 and len(rig.vertices) == 10741 and len(rig.faces) == 19528
    for name in ("Hips_01", "LeftUpLeg_02", "RightFoot_08", "Spine3_013", "LeftForeArm_016", "Neck1_023",
                 "Head_024", "EyeL_00", "Tail_026", "TailEnd_032"):
        rig.joint(name)
    assert (rig.parents >= -1).all() and (rig.parents == -1).sum() == 1
    assert np.allclose(rig.weights.sum(1), 1.0, atol=1e-5) and (rig.weights >= 0).all()
    assert set(rig.prim_material) == {"anisotropic1", "anisotropic2", "anisotropic3", "blinn1"}
    assert np.isfinite(rig.uv).all() and rig.uv.min() > -0.01 and rig.uv.max() < 1.01
    # eyes (iris and cornea) are carried by the head and eye joints only
    head_ids = {rig.joint(n) for n in ("Head_024", "EyeL_00", "EyeR_025")}
    for prim, material in enumerate(rig.prim_material):
        if material in ("anisotropic3", "blinn1"):
            s, c = rig.prim_start[prim], rig.prim_count[prim]
            used = set(rig.joints[s:s + c][rig.weights[s:s + c] > 1e-6].tolist())
            assert used <= head_ids, (material, used)


def test_rig_has_real_cat_proportions():
    rig = load_rig()
    v = rig.vertices
    tail = {rig.joint(n) for n in rig.joint_names if "Tail" in n}
    body = v[(np.isin(rig.joints, list(tail)) * rig.weights).sum(1) < 0.5]
    assert body[:, 0].max() - body[:, 0].min() == pytest.approx(0.46, abs=1e-6)  # nose to rump
    assert abs(v[:, 2].min()) < 1e-6  # paws on the floor
    shoulders = v[(v[:, 0] > -0.02) & (v[:, 0] < 0.10)]  # over the front legs, behind the head
    assert 0.23 <= shoulders[:, 2].max() <= 0.28
    head = rig.bind_pos[rig.joint("Head_024")]
    assert head[0] > 0.1 and abs(head[1]) < 0.01  # faces +x, centred


def test_manifest_records_the_licence_and_the_exact_source():
    m = json.loads((CAT_DIR / "manifest.json").read_text(encoding="utf-8"))
    assert m["license"].startswith("Creative Commons Attribution 4.0") and m["license_url"].endswith("/by/4.0/")
    assert m["creator"] == "Vr-cvantorium" and "a503ae2a7bdd43ada7f3bea3ae0a523f" in m["source_url"]
    src = (CAT_DIR / "source" / "cat_vr-cvantorium.glb").read_bytes()
    assert hashlib.sha256(src).hexdigest() == m["source_sha256"] and len(src) == m["source_bytes"]
    assert "modified" in m and m["vertices"] == 10741 and m["joints"] == 34


def test_mujoco_skinning_reproduces_the_bind_pose_exactly():
    m, d = _model()
    mujoco.mj_forward(m, d)
    assert m.nskin == len(load_rig().prim_material)
    assert np.allclose(_skin_vertices(m, d), load_rig().vertices, atol=2e-5)


def test_posing_moves_the_skin_with_its_bones():
    rig = load_rig()
    m, d = _model()
    sk = Skeleton(m, "cat0_")
    yaw = np.radians(90)
    R = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    sk.pose(np.array([1.0, 2.0, 0.0]), R)  # whole cat moved and turned: a rigid motion
    sk.write(d)
    mujoco.mj_forward(m, d)
    assert np.allclose(_skin_vertices(m, d), rig.vertices @ R.T + [1.0, 2.0, 0.0], atol=5e-5)
    # bend the tail at its base: tail vertices move, head vertices do not
    q = np.zeros(4)
    mujoco.mju_axisAngle2Quat(q, np.array([0.0, 0.0, 1.0]), np.radians(40))
    bend = np.zeros(9)
    mujoco.mju_quat2Mat(bend, q)
    sk.pose(np.zeros(3), np.eye(3), {rig.joint("Tail_026"): bend.reshape(3, 3)})
    sk.write(d)
    mujoco.mj_forward(m, d)
    moved = np.linalg.norm(_skin_vertices(m, d) - rig.vertices, axis=1)
    tail_w = (rig.joints == rig.joint("TailEnd_032")).any(1)
    head_w = (rig.joints[:, 0] == rig.joint("Head_024")) & (rig.weights[:, 0] > 0.99)
    assert moved[tail_w].min() > 0.05 and moved[head_w].max() < 1e-5
