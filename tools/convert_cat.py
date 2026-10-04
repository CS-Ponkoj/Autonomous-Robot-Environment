"""Convert the rigged cat model (glTF binary) into the simulator's cat rig: bind-pose vertices,
texture coordinates, faces per material, skin weights, and the joint hierarchy, in metres with
z up and the cat facing +x. Development-only (needs requirements-dev.txt: pygltflib).

    .venv\\Scripts\\python tools\\convert_cat.py

Source: "Cat" by Vr-cvantorium, https://sketchfab.com/3d-models/cat-a503ae2a7bdd43ada7f3bea3ae0a523f,
licensed CC BY 4.0 (http://creativecommons.org/licenses/by/4.0/). The converted rig and the
coat textures made from it are modified versions of that model and stay under CC BY 4.0.
"""

from __future__ import annotations

import hashlib
import io
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CAT_DIR = ROOT / "robot_env" / "assets" / "cat"
SOURCE = CAT_DIR / "source" / "cat_vr-cvantorium.glb"
RIG = CAT_DIR / "cat_rig.npz"
MANIFEST = CAT_DIR / "manifest.json"
SOURCE_SHA256 = "ce9a61ee8d9c0edb49834d23e1b3f3d586c9eb19997f122fabdf3d4d61a7d78d"
BODY_LENGTH = 0.46  # m, nose to rump without the tail (a typical adult domestic cat)
SOURCE_INFO = {
    "title": "Cat",
    "creator": "Vr-cvantorium",
    "creator_url": "https://sketchfab.com/Vr-cvantorium",
    "source_url": "https://sketchfab.com/3d-models/cat-a503ae2a7bdd43ada7f3bea3ae0a523f",
    "obtained_from": "Objaverse 1.0 mirror, https://huggingface.co/datasets/allenai/objaverse "
                     "(glbs/000-070/a503ae2a7bdd43ada7f3bea3ae0a523f.glb)",
    "license": "Creative Commons Attribution 4.0 International (CC BY 4.0)",
    "license_url": "http://creativecommons.org/licenses/by/4.0/",
    "modified": "Converted to a MuJoCo skin rig (axes, scale, bind data), coat textures recoloured "
                "and repainted; the original model is unchanged in source/.",
}


def _accessor(g, blob: bytes, index: int) -> np.ndarray:
    a = g.accessors[index]
    bv = g.bufferViews[a.bufferView]
    comps = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}[a.type]
    dtype = np.dtype({5126: np.float32, 5125: np.uint32, 5123: np.uint16, 5121: np.uint8}[a.componentType])
    start = (bv.byteOffset or 0) + (a.byteOffset or 0)
    stride = bv.byteStride or comps * dtype.itemsize
    rows = [np.frombuffer(blob, dtype=dtype, count=comps, offset=start + k * stride) for k in range(a.count)] \
        if stride != comps * dtype.itemsize else [np.frombuffer(blob, dtype=dtype, count=comps * a.count, offset=start)]
    return np.concatenate(rows).reshape(a.count, comps)


def _node_local(node) -> np.ndarray:
    if node.matrix:
        return np.array(node.matrix, float).reshape(4, 4).T
    x, y, z, w = node.rotation or [0, 0, 0, 1]
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                  [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                  [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    T = np.eye(4)
    T[:3, :3] = R * np.array(node.scale or [1, 1, 1])
    T[:3, 3] = node.translation or [0, 0, 0]
    return T


def load(path: Path = SOURCE) -> dict:
    """Read the glTF skin: bind-pose vertices (skeleton space), weights, UVs, faces, materials,
    joints with parents and inverse bind matrices, and the base-colour images."""
    from pygltflib import GLTF2
    g = GLTF2().load(str(path))
    blob = g.binary_blob()
    if len(g.skins) != 1:
        raise ValueError(f"expected one skin, found {len(g.skins)}")
    skin = g.skins[0]
    joints = list(skin.joints)
    parent_node = {c: i for i, n in enumerate(g.nodes) for c in (n.children or [])}
    parents = [joints.index(parent_node[j]) if parent_node.get(j) in joints else -1 for j in joints]
    ibm = _accessor(g, blob, skin.inverseBindMatrices).astype(float).reshape(-1, 4, 4).transpose(0, 2, 1)
    prims = []
    for node in g.nodes:
        if node.mesh is None:
            continue
        for p in g.meshes[node.mesh].primitives:
            if p.attributes.JOINTS_0 is None:
                raise ValueError("every primitive must be skinned")
            mat = g.materials[p.material]
            pbr = mat.pbrMetallicRoughness
            image = None
            if pbr is not None and pbr.baseColorTexture is not None:
                im = g.images[g.textures[pbr.baseColorTexture.index].source]
                bv = g.bufferViews[im.bufferView]
                image = blob[bv.byteOffset:bv.byteOffset + bv.byteLength]
            prims.append({
                "material": mat.name, "image": image,
                "pos": _accessor(g, blob, p.attributes.POSITION).astype(float),
                "uv": _accessor(g, blob, p.attributes.TEXCOORD_0).astype(float),
                "normal": _accessor(g, blob, p.attributes.NORMAL).astype(float),
                "faces": _accessor(g, blob, p.indices).astype(np.int64).reshape(-1, 3),
                "joints": _accessor(g, blob, p.attributes.JOINTS_0).astype(np.int64),
                "weights": _accessor(g, blob, p.attributes.WEIGHTS_0).astype(float),
            })
    return {"joint_names": [g.nodes[j].name for j in joints], "parents": parents, "ibm": ibm, "prims": prims}


def convert(src: dict) -> dict:
    """Axes (glTF y-up to z-up), facing +x, scale to BODY_LENGTH, feet on z = 0; bind poses of
    the joints as positions and rotation matrices in the same frame."""
    A = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float)  # y-up -> z-up
    names = src["joint_names"]
    bind = [np.linalg.inv(m) for m in src["ibm"]]  # joint world transforms at bind
    jpos = np.array([A @ b[:3, 3] for b in bind])
    head = jpos[[i for i, n in enumerate(names) if n.endswith("Head_024")][0]]
    hips = jpos[[i for i, n in enumerate(names) if n.endswith("Hips_01")][0]]
    fwd = head - hips
    yaw = np.arctan2(fwd[1], fwd[0])
    c, s = np.cos(-yaw), np.sin(-yaw)
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    M = Rz @ A
    verts = np.vstack([p["pos"] for p in src["prims"]]) @ M.T
    tail = np.array([i for i, n in enumerate(names) if "Tail" in n])
    J = np.vstack([p["joints"] for p in src["prims"]])
    W = np.vstack([p["weights"] for p in src["prims"]])
    tail_w = (np.isin(J, tail) * W).sum(1)
    body = verts[tail_w < 0.5]
    scale = BODY_LENGTH / (body[:, 0].max() - body[:, 0].min())
    shift = np.array([-(body[:, 0].max() + body[:, 0].min()) / 2, -(body[:, 1].max() + body[:, 1].min()) / 2,
                      -verts[:, 2].min()])
    out_v = (verts + shift) * scale
    bind_pos, bind_rot = [], []
    for b in bind:
        R = M @ b[:3, :3] @ M.T  # joint frame expressed in the converted axes
        R /= np.linalg.norm(R, axis=0)
        bind_pos.append((M @ b[:3, 3] + shift) * scale)
        bind_rot.append(R)
    counts = [len(p["pos"]) for p in src["prims"]]
    starts = np.cumsum([0] + counts[:-1])
    w = W / W.sum(1, keepdims=True)
    return {
        "vertices": out_v.astype(np.float32),
        "normals": (np.vstack([p["normal"] for p in src["prims"]]) @ M.T).astype(np.float32),
        "uv": np.vstack([p["uv"] for p in src["prims"]]).astype(np.float32),
        "joints": J.astype(np.int16), "weights": w.astype(np.float32),
        "prim_start": np.array(starts, np.int64), "prim_count": np.array(counts, np.int64),
        "faces": np.vstack([p["faces"] + st for p, st in zip(src["prims"], starts)]).astype(np.int32),
        "face_prim": np.concatenate([np.full(len(p["faces"]), i) for i, p in enumerate(src["prims"])]).astype(np.int16),
        "prim_material": np.array([p["material"] for p in src["prims"]]),
        "joint_names": np.array(names), "parents": np.array(src["parents"], np.int16),
        "bind_pos": np.array(bind_pos, np.float64), "bind_rot": np.array(bind_rot, np.float64),
        "scale": np.array(scale),
    }


def main() -> int:
    data = SOURCE.read_bytes()
    sha = hashlib.sha256(data).hexdigest()
    if sha != SOURCE_SHA256:
        raise SystemExit(f"source file changed: sha256 {sha}")
    src = load(SOURCE)
    rig = convert(src)
    CAT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(RIG, **rig)
    from PIL import Image
    textures = {}
    for p in src["prims"]:
        if p["image"] is not None and p["material"] not in textures:
            name = f"cat_{p['material']}.png"
            Image.open(io.BytesIO(p["image"])).convert("RGB").save(CAT_DIR / name)
            textures[p["material"]] = name
    manifest = {
        "asset": "cat", **SOURCE_INFO,
        "source_file": str(SOURCE.relative_to(ROOT)).replace("\\", "/"), "source_sha256": sha, "source_bytes": len(data),
        "converted_file": str(RIG.relative_to(ROOT)).replace("\\", "/"), "converted_bytes": RIG.stat().st_size,
        "textures": textures, "joints": len(rig["joint_names"]), "vertices": int(len(rig["vertices"])),
        "faces": int(len(rig["faces"])), "scale_to_metres": float(rig["scale"]),
        "conversion": {"command": "python tools/convert_cat.py", "python": sys.version.split()[0],
                       "date": time.strftime("%Y-%m-%d")},
    }
    MANIFEST.write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("joints", "vertices", "faces", "converted_bytes")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
