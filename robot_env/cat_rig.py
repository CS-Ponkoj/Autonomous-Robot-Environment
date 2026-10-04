"""The realistic cat's skinned mesh in MuJoCo: per-cat bone bodies, materials, and binary skin
files (one skin per material), plus forward kinematics to pose the bones.

The rig comes from tools/convert_cat.py ("Cat" by Vr-cvantorium, CC BY 4.0; see
robot_env/assets/cat/manifest.json). Bones are mocap bodies (visual only: no geoms, no mass);
the cat's physics and lidar use the simple collision bodies in robot_env/cats.py.
"""

from __future__ import annotations

import functools
import struct
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

CAT_DIR = Path(__file__).resolve().parent / "assets" / "cat"
RIG_FILE = CAT_DIR / "cat_rig.npz"
CORNEA_RGBA = "1 1 1 0.15"  # the clear eye covers (no texture in the source)


@dataclass(frozen=True)
class Rig:
    vertices: np.ndarray  # (n, 3) bind pose, metres, cat frame (x forward, z up, feet at z 0)
    uv: np.ndarray
    faces: np.ndarray
    face_prim: np.ndarray
    prim_start: np.ndarray
    prim_count: np.ndarray
    prim_material: tuple
    joints: np.ndarray  # (n, 4) joint indices per vertex
    weights: np.ndarray  # (n, 4) normalized
    joint_names: tuple
    parents: np.ndarray  # parent joint index, -1 for the root
    bind_pos: np.ndarray  # (j, 3) joint positions at bind, cat frame
    bind_rot: np.ndarray  # (j, 3, 3) joint orientations at bind, cat frame

    def joint(self, suffix: str) -> int:
        """Index of the joint whose name ends with `suffix` (e.g. "Head_024")."""
        for i, n in enumerate(self.joint_names):
            if n.endswith(suffix):
                return i
        raise KeyError(suffix)


@functools.cache
def load_rig(path: Path = RIG_FILE) -> Rig:
    r = np.load(path)
    return Rig(r["vertices"].astype(np.float64), r["uv"], r["faces"], r["face_prim"], r["prim_start"],
               r["prim_count"], tuple(str(m) for m in r["prim_material"]), r["joints"].astype(np.int64),
               r["weights"].astype(np.float64), tuple(str(n) for n in r["joint_names"]), r["parents"].astype(np.int64),
               r["bind_pos"], r["bind_rot"])


def _quat(R: np.ndarray) -> np.ndarray:
    q = np.empty(4)
    mujoco.mju_mat2Quat(q, np.ascontiguousarray(R).reshape(-1))
    return q


def bone_body(prefix: str, j: int) -> str:
    return f"{prefix}b{j}"


def skin_bytes(rig: Rig, prim: int, prefix: str) -> bytes:
    """One material's skin in MuJoCo's binary SKN format: counts, vertices, texture
    coordinates, faces, then per bone its body name, bind pose, and weighted vertices."""
    start, count = int(rig.prim_start[prim]), int(rig.prim_count[prim])
    faces = rig.faces[rig.face_prim == prim] - start
    v = rig.vertices[start:start + count]
    uv = rig.uv[start:start + count]
    jj, ww = rig.joints[start:start + count], rig.weights[start:start + count]
    bones = []
    for j in range(len(rig.joint_names)):
        ids, wts = [], []
        for slot in range(4):
            sel = np.flatnonzero((jj[:, slot] == j) & (ww[:, slot] > 1e-6))
            ids.append(sel)
            wts.append(ww[sel, slot])
        ids, wts = np.concatenate(ids), np.concatenate(wts)
        if len(ids):
            bones.append((j, ids, wts))
    out = [struct.pack("<4i", count, count, len(faces), len(bones)),
           v.astype("<f4").tobytes(), uv.astype("<f4").tobytes(), faces.astype("<i4").tobytes()]
    for j, ids, wts in bones:
        name = bone_body(prefix, j).encode("ascii")
        if len(name) > 39:
            raise ValueError("bone body name too long for the skin file")
        out += [name.ljust(40, b"\0"), rig.bind_pos[j].astype("<f4").tobytes(), _quat(rig.bind_rot[j]).astype("<f4").tobytes(),
                struct.pack("<i", len(ids)), ids.astype("<i4").tobytes(), wts.astype("<f4").tobytes()]
    return b"".join(out)


def model_parts(prefix: str, textures: dict[str, str], hidden: bool = False) -> tuple[str, str, str, dict]:
    """MJCF fragments for one cat: (assets, bodies for <worldbody>, skins for <deformable>, files
    for MuJoCo's in-memory asset table). `textures` maps a source material name to a PNG file in
    robot_env/assets/cat (the coat); materials without a texture are the clear corneas."""
    rig = load_rig()
    assets, skins, files = [], [], {}
    for prim, material in enumerate(rig.prim_material):
        mat = f"{prefix}m{prim}"
        tex = textures.get(material)
        if tex is not None:
            files.setdefault(tex, (CAT_DIR / tex).read_bytes())
            assets.append(f'<texture name="{prefix}t{prim}" type="2d" file="{tex}"/>'
                          f'<material name="{mat}" texture="{prefix}t{prim}" specular="0.1" shininess="0.05"/>')
        else:
            assets.append(f'<material name="{mat}" rgba="{CORNEA_RGBA}" specular="0.9" shininess="0.9"/>')
        skin_file = f"{prefix}skin{prim}.skn"
        files[skin_file] = skin_bytes(rig, prim, prefix)
        skins.append(f'<skin name="{prefix}skin{prim}" file="{skin_file}" material="{mat}" group="2"/>')
    bodies = "".join(
        f'<body name="{bone_body(prefix, j)}" mocap="true" pos="{p[0]:.5f} {p[1]:.5f} {p[2]:.5f}" '
        f'quat="{" ".join(f"{x:.6f}" for x in _quat(R))}"/>'
        for j, (p, R) in enumerate(zip(rig.bind_pos, rig.bind_rot)))
    return "".join(assets), bodies, "".join(skins), files


class Skeleton:
    """Forward kinematics: local joint rotations (relative to the bind pose) and a root pose give
    every bone's world pose, which is written to the bone mocap bodies."""

    def __init__(self, model: mujoco.MjModel, prefix: str):
        self.rig = rig = load_rig()
        n = len(rig.joint_names)
        self.mocap = np.array([model.body_mocapid[model.body(bone_body(prefix, j)).id] for j in range(n)])
        order, seen = [], set()
        while len(order) < n:  # parents before children
            for j in range(n):
                if j not in seen and (rig.parents[j] < 0 or rig.parents[j] in seen):
                    order.append(j)
                    seen.add(j)
        self.order = order
        # each joint's bind pose relative to its parent
        self.rel_pos = np.zeros((n, 3))
        self.rel_rot = np.tile(np.eye(3), (n, 1, 1))
        for j in range(n):
            p = rig.parents[j]
            if p >= 0:
                self.rel_pos[j] = rig.bind_rot[p].T @ (rig.bind_pos[j] - rig.bind_pos[p])
                self.rel_rot[j] = rig.bind_rot[p].T @ rig.bind_rot[j]
        self.pos = rig.bind_pos.copy()
        self.rot = rig.bind_rot.copy()

    def pose(self, root_pos, root_rot, local: dict[int, np.ndarray] | None = None) -> None:
        """World poses for a cat whose frame (x forward, z up) is at root_pos with root_rot;
        `local` adds rotations (3x3, in each joint's own frame) to the bind pose."""
        rig, local = self.rig, local or {}
        for j in self.order:
            p = rig.parents[j]
            extra = local.get(j)
            if p < 0:
                rot = root_rot @ rig.bind_rot[j]
                self.pos[j] = root_pos + root_rot @ rig.bind_pos[j]
            else:
                rot = self.rot[p] @ self.rel_rot[j]
                self.pos[j] = self.pos[p] + self.rot[p] @ self.rel_pos[j]
            self.rot[j] = rot @ extra if extra is not None else rot

    def write(self, data: mujoco.MjData) -> None:
        data.mocap_pos[self.mocap] = self.pos
        for j, m in enumerate(self.mocap):
            mujoco.mju_mat2Quat(data.mocap_quat[m], self.rot[j].reshape(-1))
