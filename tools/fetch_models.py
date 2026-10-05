"""Download CC0 furniture models from Poly Haven and convert them for MuJoCo's classic renderer.
Development tool: the converted files are committed under robot_env/assets/furniture/.

    .venv\\Scripts\\python tools\\fetch_models.py             (every model in MODELS)
    .venv\\Scripts\\python tools\\fetch_models.py sofa_03      (only these)
    .venv\\Scripts\\python tools\\fetch_models.py --check     (convert again elsewhere: byte-identical?)

For each model the 1k glTF and its files are downloaded once into the model cache (never
published) and checked against Poly Haven's published MD5 sums; their SHA-256 sums are recorded.
The scene is flattened, turned from glTF's y-up to z-up, scaled to Poly Haven's listed real size
when it was authored at another scale, centred on x/y with its base at z = 0, and split by
material. Zero-area triangles are dropped and the model is simplified to its face cap (the total,
over every part). Each part becomes a binary MuJoCo mesh (.msh: positions, normals, texture
coordinates, faces) and a PNG colour texture with the model's ambient occlusion baked in (the
classic renderer has no AO or normal maps).

Gates (a model that fails one is not written): every output vertex lies within FIDELITY_M of the
source surface, the source surface lies within COVERAGE_MAX_M of the output (no holes), and the
output bounds match the source bounds; normals are finite and non-zero;
faces on edges shared by two triangles are wound consistently (or the model is listed in
WINDING_OK with the reason). <id>.json records the parts, their shine (from roughness and
metalness), the bounds, the source, the licence, the hashes, and the checks; MANIFEST.json lists
every committed file with its SHA-256 and size.
"""

from __future__ import annotations

import hashlib
import io
import json
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from importlib import metadata
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "robot_env" / "assets" / "furniture"
API = "https://api.polyhaven.com"
AGENT = {"User-Agent": "autonomous-robot-environment-dev/1.0"}

AO_STRENGTH = 0.8
FIDELITY_M = 0.01  # largest distance from an output vertex to the source surface
COVERAGE_MAX_M = 0.02  # largest distance from the source surface to the output (holes, tears)
COVERAGE_999_M = 0.01  # the same at the 99.9th percentile
EDGE_STEP_M = 0.005  # coverage samples along every source edge
BOUNDS_M = 0.01  # largest change of the model's bounds by the conversion (simplification rounds extremities)

# model id -> (texture size, face cap over the whole model[, {material name part: its own cap}]).
# Only models placed in the world.
MODELS = {
    "metal_office_desk": (512, 8000),
    "steel_frame_shelves_01": (512, 6000),
    "potted_plant_01": (512, 43000, {"_pot": 25000, "_leaves": 18000}),
    "wall_clock": (256, 3658),  # its own face count: not simplified (it is small)
}
WINDING_OK: dict[str, str] = {}  # model id -> why its inconsistent winding is accepted


def cache_dir() -> Path:
    """The model cache: _private/model_cache of the main checkout (shared by git worktrees)."""
    try:
        common = subprocess.run(["git", "rev-parse", "--path-format=absolute", "--git-common-dir"], cwd=ROOT,
                                capture_output=True, text=True, timeout=10, check=True).stdout.strip()
        return Path(common).parent / "_private" / "model_cache"
    except (OSError, subprocess.SubprocessError):
        return ROOT / "_private" / "model_cache"


def _get(url: str) -> bytes:
    with urllib.request.urlopen(urllib.request.Request(url, headers=AGENT), timeout=60) as r:
        return r.read()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def download(model_id: str) -> tuple[Path, dict]:
    """The model's 1k glTF (with its files) in the cache, checked against Poly Haven's MD5 sums
    (also when it was cached earlier); returns (gltf path, info with the files' SHA-256 sums)."""
    folder = cache_dir() / model_id
    info_path = folder / "info.json"
    files = json.loads(_get(f"{API}/files/{model_id}"))
    gltf = files["gltf"]["1k"]["gltf"]
    entries = {Path(gltf["url"]).name: gltf, **gltf["include"]}
    if info_path.exists():
        info = json.loads(info_path.read_text(encoding="utf-8"))
    else:
        info = json.loads(_get(f"{API}/info/{model_id}"))
    folder.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for rel, entry in sorted(entries.items()):
        target = folder / rel
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(_get(entry["url"]))
        data = target.read_bytes()
        if hashlib.md5(data).hexdigest() != entry["md5"]:
            raise RuntimeError(f"{model_id}: {rel} does not match Poly Haven's MD5")
        hashes[rel] = _sha256(data)
    info["_gltf"] = Path(gltf["url"]).name
    info["_source"] = gltf["url"]
    info["_files"] = hashes
    info_path.write_text(json.dumps(info, sort_keys=True), encoding="utf-8")
    return folder / info["_gltf"], info


def _arm_texture(gltf_path: Path, material_name: str) -> Image.Image | None:
    """The material's packed AO/roughness/metal map (Poly Haven names it *_arm_1k.jpg)."""
    for p in sorted((gltf_path.parent / "textures").glob("*_arm_1k.jpg")):
        stem = p.name.replace("_arm_1k.jpg", "")
        if material_name and stem.endswith(material_name):
            return Image.open(p).convert("RGB")
    return None


def msh_bytes(v: np.ndarray, n: np.ndarray, uv: np.ndarray, f: np.ndarray) -> bytes:
    """MuJoCo's binary mesh: int32 counts (vertices, normals, texcoords, faces), then float32
    positions, normals and texcoords, then int32 faces. Texture rows: MuJoCo reads .msh texture
    coordinates as given (it flips OBJ's itself), so glTF's top-down v becomes 1 - v here."""
    uv = np.column_stack([uv[:, 0], 1.0 - uv[:, 1]])
    head = np.array([len(v), len(n), len(uv), len(f)], dtype="<i4").tobytes()
    return (head + np.ascontiguousarray(v, dtype="<f4").tobytes() + np.ascontiguousarray(n, dtype="<f4").tobytes()
            + np.ascontiguousarray(uv, dtype="<f4").tobytes() + np.ascontiguousarray(f, dtype="<i4").tobytes())


def read_msh(data: bytes) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(positions, normals, texcoords as written, faces) of a .msh file."""
    nv, nn, nt, nf = np.frombuffer(data[:16], dtype="<i4")
    o = 16
    out = []
    for count, width, dtype in ((nv, 3, "<f4"), (nn, 3, "<f4"), (nt, 2, "<f4"), (nf, 3, "<i4")):
        size = int(count) * width * 4
        out.append(np.frombuffer(data[o:o + size], dtype=dtype).reshape(-1, width))
        o += size
    if o != len(data):
        raise ValueError("trailing bytes in a .msh file")
    return tuple(out)


def _clean(v, f, uv):
    """Drop zero-area triangles and unreferenced vertices."""
    area2 = np.linalg.norm(np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]]), axis=1)
    f = f[area2 > 1e-12]
    used = np.unique(f)
    remap = np.full(len(v), -1, np.int64)
    remap[used] = np.arange(len(used))
    return v[used], remap[f], uv[used], int(len(area2) - len(f))


def _visible_vertices(m) -> np.ndarray:
    """Vertices of the triangles with area (zero-area ones are invisible and dropped by _clean;
    one plant pot has such slivers 14 mm below its base)."""
    v, f = np.asarray(m.vertices, float), np.asarray(m.faces, np.int64)
    area2 = np.linalg.norm(np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]]), axis=1)
    return v[np.unique(f[area2 > 1e-12])]


def _weld(v: np.ndarray, f: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Merge vertices at the same position (texture seams duplicate them), dropping faces that
    collapse."""
    _, idx, inv = np.unique(np.round(v / 1e-7).astype(np.int64), axis=0, return_index=True, return_inverse=True)
    wf = inv.ravel()[f]
    keep = (wf[:, 0] != wf[:, 1]) & (wf[:, 1] != wf[:, 2]) & (wf[:, 0] != wf[:, 2])
    return v[idx], wf[keep]


def _closest_on_triangles(p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Closest points to p on triangles abc (all (..., 3)): Ericson's region test, vectorized."""
    ab, ac, ap = b - a, c - a, p - a
    d1, d2 = (ab * ap).sum(-1), (ac * ap).sum(-1)
    bp = p - b
    d3, d4 = (ab * bp).sum(-1), (ac * bp).sum(-1)
    cp = p - c
    d5, d6 = (ab * cp).sum(-1), (ac * cp).sum(-1)
    va = d3 * d6 - d5 * d4
    vb = d5 * d2 - d1 * d6
    vc = d1 * d4 - d3 * d2
    den = va + vb + vc
    den = np.where(np.abs(den) < 1e-30, 1e-30, den)
    out = a + ab * (vb / den)[..., None] + ac * (vc / den)[..., None]  # inside the face
    with np.errstate(divide="ignore", invalid="ignore"):
        e_ab = (vc <= 0) & (d1 >= 0) & (d3 <= 0)
        out = np.where(e_ab[..., None], a + ab * np.nan_to_num(d1 / (d1 - d3))[..., None], out)
        e_ac = (vb <= 0) & (d2 >= 0) & (d6 <= 0)
        out = np.where(e_ac[..., None], a + ac * np.nan_to_num(d2 / (d2 - d6))[..., None], out)
        e_bc = (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0)
        t = np.nan_to_num((d4 - d3) / ((d4 - d3) + (d5 - d6)))
        out = np.where(e_bc[..., None], b + (c - b) * t[..., None], out)
    out = np.where(((d1 <= 0) & (d2 <= 0))[..., None], a, out)
    out = np.where(((d3 >= 0) & (d4 <= d3))[..., None], b, out)
    out = np.where(((d6 >= 0) & (d5 <= d6))[..., None], c, out)
    return out


def _transfer_uv(v, f, uv, nv, nf, k: int = 8) -> np.ndarray:
    """Texture coordinates (faces, 3, 2) for simplified triangles: each output triangle takes the
    source triangle nearest its centre, and each corner the UV interpolated (barycentric, in that
    triangle's plane) at the corner's nearest point of it. Seams stay seams: a triangle never mixes
    two UV charts, and no UV leaves its chart."""
    from scipy.spatial import cKDTree
    tri = v[f]
    cent = tri.mean(1)
    q = nv[nf].mean(1)
    _, cand = cKDTree(cent).query(q, k=min(k, len(f)))
    cand = np.asarray(cand).reshape(len(q), -1)
    a, b, c = tri[cand, 0], tri[cand, 1], tri[cand, 2]
    d = np.linalg.norm(_closest_on_triangles(q[:, None, :], a, b, c) - q[:, None, :], axis=-1)
    best = cand[np.arange(len(q)), np.argmin(d, axis=1)]
    a, b, c = tri[best, 0][:, None], tri[best, 1][:, None], tri[best, 2][:, None]
    # each corner taken to the nearest point of that triangle, so its UV stays inside the
    # triangle's chart (extrapolating past a seam sampled another part of the texture)
    p = _closest_on_triangles(nv[nf], a, b, c)  # (m, 3 corners, 3)
    v0, v1, v2 = b - a, c - a, p - a
    d00, d01, d11 = (v0 * v0).sum(-1), (v0 * v1).sum(-1), (v1 * v1).sum(-1)
    d20, d21 = (v2 * v0).sum(-1), (v2 * v1).sum(-1)
    den = d00 * d11 - d01 * d01
    den = np.where(np.abs(den) < 1e-30, 1e-30, den)
    wb = (d11 * d20 - d01 * d21) / den
    wc = (d00 * d21 - d01 * d20) / den
    wa = 1.0 - wb - wc
    t = uv[f[best]]  # (m, 3, 2): the source triangle's UVs
    return wa[..., None] * t[:, None, 0] + wb[..., None] * t[:, None, 1] + wc[..., None] * t[:, None, 2]


def _simplify(v, f, uv, target):
    """Quadric simplification to `target` faces. The seams are welded first (a seam split into
    two open borders tears when simplified: one planter stand got a 4 cm hole); the texture
    coordinates then come from the source triangles (_transfer_uv), and smooth normals from the
    welded result. Returns (vertices, faces, uv, normals), or the input with normals None."""
    import fast_simplification
    if target >= len(f):
        return v, f, uv, None
    wv, wf = _weld(v, f)
    nv, nf = fast_simplification.simplify(wv, wf, target_count=int(min(target, len(wf) - 1)))
    nv, nf, _, _ = _clean(nv, nf, np.zeros((len(nv), 2)))
    nn = _normals(nv, nf)
    cuv = _transfer_uv(v, f, uv, nv, nf).reshape(-1, 2)
    cv, cn = nv[nf].reshape(-1, 3), nn[nf].reshape(-1, 3)
    key = np.column_stack([np.round(cv / 1e-7), np.round(cuv / 1e-6)]).astype(np.int64)
    _, idx, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    return cv[idx], inv.ravel().reshape(-1, 3), cuv[idx], cn[idx]


def _coverage(src_meshes, out_meshes) -> tuple[float, float]:
    """(largest, 99.9th percentile) distance from the source surface to the output surface: a hole
    or a torn part shows here, not in the vertex distance. Sampled (deterministic): every source
    vertex, points every EDGE_STEP_M along every source edge, and seeded samples of the source
    faces; each point is measured exactly to the output triangles that the 16 nearest
    output-surface samples lie on (an upper bound on that point's true distance)."""
    import trimesh
    from scipy.spatial import cKDTree
    src = trimesh.util.concatenate(src_meshes)
    area2 = np.linalg.norm(np.cross(src.vertices[src.faces[:, 1]] - src.vertices[src.faces[:, 0]],
                                    src.vertices[src.faces[:, 2]] - src.vertices[src.faces[:, 0]]), axis=1)
    src = trimesh.Trimesh(src.vertices, src.faces[area2 > 1e-12], process=False)  # what is visible
    verts, faces, off = [], [], 0
    for v, f in out_meshes:
        verts.append(v)
        faces.append(f + off)
        off += len(v)
    out = trimesh.Trimesh(np.vstack(verts), np.vstack(faces), process=False)
    sp = trimesh.sample.sample_surface(src, int(min(200_000, max(20_000, src.area / 5e-5))), seed=0)[0]
    edges = src.vertices[src.edges_unique]
    steps = np.maximum(1, np.ceil(np.linalg.norm(edges[:, 1] - edges[:, 0], axis=1) / EDGE_STEP_M)).astype(int)
    which = np.repeat(np.arange(len(edges)), steps)
    t = (np.arange(steps.sum()) - np.repeat(np.cumsum(steps) - steps, steps) + 0.5) / np.repeat(steps, steps)
    edge_pts = edges[which, 0] + (edges[which, 1] - edges[which, 0]) * t[:, None]
    sp = np.vstack([sp, src.vertices[np.unique(src.faces)], edge_pts])
    op, of = trimesh.sample.sample_surface(out, int(min(600_000, max(60_000, out.area / 2e-5))), seed=1)
    op = np.vstack([op, out.vertices])
    of = np.concatenate([of, np.array([fs[0] for fs in out.vertex_faces], dtype=np.int64)])
    _, nb = cKDTree(op).query(sp, k=16)
    cand = of[nb]  # (n, 16) triangles
    tri = out.vertices[out.faces]
    best = np.full(len(sp), np.inf)
    for j in range(cand.shape[1]):
        t = tri[cand[:, j]]
        d = np.linalg.norm(_closest_on_triangles(sp, t[:, 0], t[:, 1], t[:, 2]) - sp, axis=1)
        best = np.minimum(best, d)
    return float(best.max()), float(np.percentile(best, 99.9))


def _normals(v: np.ndarray, f: np.ndarray) -> np.ndarray:
    """Area-weighted vertex normals. Where they cancel (two-sided sheets: a face and its reverse
    share the vertices), the vertex takes the normal of its largest adjacent face."""
    fn = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    n = np.zeros_like(v)
    for c in range(3):
        np.add.at(n, f[:, c], fn)
    length = np.linalg.norm(n, axis=1)
    weak = length < 1e-12 * np.maximum(1.0, np.linalg.norm(fn, axis=1).max())
    if np.any(weak):
        area = np.linalg.norm(fn, axis=1)
        best = np.full(len(v), -1.0)
        pick = np.zeros((len(v), 3))
        for c in range(3):
            better = area > best[f[:, c]]
            idx = f[better, c]
            order = np.argsort(area[better], kind="stable")  # the largest face is written last
            best[idx[order]] = area[better][order]
            pick[idx[order]] = fn[better][order]
        n[weak] = pick[weak]
        length = np.linalg.norm(n, axis=1)
    return n / np.maximum(length, 1e-30)[:, None]


def _winding_errors(f: np.ndarray) -> tuple[int, int]:
    """(edges shared by exactly two faces, those whose two faces run them the same way)."""
    e = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    key = np.sort(e, axis=1)
    forward = e[:, 0] < e[:, 1]
    _, inv, counts = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    inv = inv.ravel()
    two = counts[inv] == 2
    order = np.argsort(inv[two], kind="stable")
    fw = forward[two][order].reshape(-1, 2)
    return int(len(fw)), int(np.sum(fw[:, 0] == fw[:, 1]))


def _fidelity(src_meshes, v_out: np.ndarray) -> float:
    """Largest distance from an output vertex to the source surface (source vertices plus a
    dense, seeded sample of its triangles)."""
    import trimesh
    from scipy.spatial import cKDTree
    pts = [np.vstack([m.vertices for m in src_meshes])]
    for m in src_meshes:
        count = int(min(400_000, max(2_000, m.area / 1e-5)))
        pts.append(trimesh.sample.sample_surface(m, count, seed=0)[0])
    return float(cKDTree(np.vstack(pts)).query(v_out)[0].max())


def convert(model_id: str, size: int, cap: int, out: Path, part_caps: dict | None = None) -> dict:
    """Convert one model so its faces stay within `cap` (the shares of small parts round up,
    so the budget handed out is tightened until the total fits)."""
    budget = cap
    for _ in range(8):
        record = _convert(model_id, size, cap, budget, out, part_caps)
        faces = record["checks"]["faces"]
        if faces <= cap:
            return record
        budget = int(budget * cap / faces * 0.98)
    raise RuntimeError(f"{model_id}: cannot simplify to {cap} faces ({faces} reached)")


def _convert(model_id: str, size: int, cap: int, budget: int, out: Path, part_caps: dict | None = None) -> dict:
    import trimesh

    gltf_path, info = download(model_id)
    scene = trimesh.load(gltf_path, force="scene")
    y_up_to_z_up = trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])
    parts: dict[str, list] = {}
    for mesh in scene.dump():
        if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
            continue
        mesh.apply_transform(y_up_to_z_up)
        mat = getattr(mesh.visual, "material", None)
        key = getattr(mat, "name", None) or f"part{len(parts)}"
        parts.setdefault(key, []).append(mesh)
    if not parts:
        raise RuntimeError(f"{model_id}: no meshes")
    allv = np.vstack([_visible_vertices(m) for ms in parts.values() for m in ms])
    lo, hi = allv.min(0), allv.max(0)
    scale = 1.0
    listed = info.get("dimensions")  # the real size Poly Haven lists, in millimetres
    if listed:
        factor = max(listed) / 1000.0 / float((hi - lo).max())
        if abs(factor - 1.0) > 0.05:  # authored at another scale (one shelf model is 10x): fix it
            scale = factor
    shift = np.array([-(lo[0] + hi[0]) / 2, -(lo[1] + hi[1]) / 2, -lo[2]]) * scale
    for ms in parts.values():
        for m in ms:
            m.apply_scale(scale)
            m.apply_translation(shift)
    src_lo, src_hi = lo * scale + shift, hi * scale + shift
    total = sum(len(m.faces) for ms in parts.values() for m in ms)
    part_total = {sub: sum(len(m.faces) for k, ms in parts.items() if sub in k for m in ms) or 1
                  for sub in (part_caps or {})}
    out_parts, removed, out_v, out_meshes = [], 0, [], []
    for k, (key, meshes) in enumerate(sorted(parts.items())):
        verts, faces, uvs, normals = [], [], [], []
        offset = 0
        for m in meshes:
            uv = getattr(m.visual, "uv", None)
            uv = np.zeros((len(m.vertices), 2)) if uv is None else np.asarray(uv, float)
            v, f, uv, dropped = _clean(np.asarray(m.vertices, float), np.asarray(m.faces, np.int64), uv)
            removed += dropped
            # every source mesh gets its share of its material's cap, or of the model's
            share = max(4, (budget * len(m.faces)) // total)
            for sub, mcap in (part_caps or {}).items():
                if sub in key:
                    share = max(4, (mcap * budget // cap) * len(m.faces) // part_total[sub])
            v, f, uv, n = _simplify(v, f, uv, share)
            if n is None:
                n = _normals(v, f)
            verts.append(v)
            faces.append(f + offset)
            uvs.append(uv)
            normals.append(n)
            offset += len(v)
        v, f, uv, n = np.vstack(verts), np.vstack(faces), np.vstack(uvs), np.vstack(normals)
        if not np.all(np.isfinite(n)) or np.any(np.linalg.norm(n, axis=1) < 0.5):
            raise RuntimeError(f"{model_id} part {k}: a normal is not finite or not unit length")
        shared, wrong = _winding_errors(_weld(v, f)[1])
        if wrong > 0.01 * max(shared, 1) and model_id not in WINDING_OK:
            raise RuntimeError(f"{model_id} part {k}: {wrong} of {shared} shared edges wound inconsistently")
        out_v.append(v)
        out_meshes.append((v, f))
        part = f"{model_id}_{k}"
        data = msh_bytes(v, n, uv, f)
        (out / f"{part}.msh").write_bytes(data)
        mat = getattr(meshes[0].visual, "material", None)
        base = getattr(mat, "baseColorTexture", None)
        factor = np.asarray(getattr(mat, "baseColorFactor", None) if getattr(mat, "baseColorFactor", None) is not None
                            else [255, 255, 255, 255], float)
        factor = factor / (255.0 if factor.max() > 1.0 else 1.0)
        rough = float(getattr(mat, "roughnessFactor", 1.0) or 1.0)
        metal = float(getattr(mat, "metallicFactor", 0.0) or 0.0)
        arm = _arm_texture(gltf_path, key)
        if arm is not None:
            a = np.asarray(arm, float) / 255.0
            rough *= float(a[..., 1].mean())
            metal *= float(a[..., 2].mean()) if metal > 0 else 0.0
        texture = None
        if base is not None:
            img = np.asarray(base.convert("RGB"), float) / 255.0 * factor[:3]
            if arm is not None:
                ao = np.asarray(arm.resize(base.size), float)[..., 0] / 255.0
                img *= (1.0 - AO_STRENGTH * (1.0 - ao))[..., None]
            im = Image.fromarray(np.clip(img * 255.0, 0, 255).astype(np.uint8)).resize((size, size), Image.LANCZOS)
            texture = f"{part}.png"
            buf = io.BytesIO()
            im.save(buf, "PNG", optimize=True)
            (out / texture).write_bytes(buf.getvalue())
        out_parts.append({
            "mesh": f"{part}.msh", "material": key, "texture": texture,
            "rgba": [round(float(c), 3) for c in factor[:3]] + [1.0],
            "specular": round(0.5 * (1.0 - rough) + 0.4 * metal, 3), "shininess": round(max(0.02, 1.0 - rough) ** 2, 3),
            "reflectance": round(0.15 * metal * (1.0 - rough), 3), "faces": int(len(f)), "vertices": int(len(v)),
            "winding": {"shared_edges": shared, "inconsistent": wrong},
        })
    allout = np.vstack(out_v)
    faces_out = sum(p["faces"] for p in out_parts)
    bounds_err = float(max(np.abs(allout.min(0) - src_lo).max(), np.abs(allout.max(0) - src_hi).max()))
    if bounds_err > BOUNDS_M:
        raise RuntimeError(f"{model_id}: conversion moved the bounds by {bounds_err * 1000:.1f} mm")
    fidelity = _fidelity([m for ms in parts.values() for m in ms], allout)
    if fidelity > FIDELITY_M:
        raise RuntimeError(f"{model_id}: an output vertex is {fidelity * 1000:.1f} mm from the source surface")
    cover_max, cover_999 = _coverage([m for ms in parts.values() for m in ms], out_meshes)
    if cover_max > COVERAGE_MAX_M or cover_999 > COVERAGE_999_M:
        raise RuntimeError(f"{model_id}: the source surface is up to {cover_max * 1000:.1f} mm "
                           f"(99.9%: {cover_999 * 1000:.1f} mm) from the output surface (a hole?)")
    lo_o, hi_o = allout.min(0), allout.max(0)
    record = {
        "id": model_id, "name": info.get("name", model_id), "authors": sorted(info.get("authors", {})),
        "license": "CC0 1.0", "source": f"https://polyhaven.com/a/{model_id}", "file": info.get("_source"),
        "source_files_sha256": info["_files"], "listed_dimensions_mm": listed, "scale": round(scale, 6),
        "size_m": [round(float(b), 4) for b in hi_o - lo_o], "parts": out_parts,
        "checks": {"source_faces": int(total), "faces": int(faces_out), "face_cap": cap,
                   "zero_area_faces_removed": int(removed), "max_vertex_distance_m": round(fidelity, 5),
                   "bounds_change_m": round(bounds_err, 5), "coverage_max_m": round(cover_max, 5),
                   "coverage_p999_m": round(cover_999, 5)},
    }
    (out / f"{model_id}.json").write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8", newline="\n")
    return record


def tool_record() -> dict:
    """The converter and the libraries it ran with."""
    names = ("numpy", "pillow", "trimesh", "fast_simplification", "scipy")
    deps = {}
    for name in names:
        try:
            deps[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            deps[name] = None
    def text_sha(path: Path) -> str:  # line endings as committed (a Windows checkout may have CRLF)
        return _sha256(path.read_bytes().replace(b"\r\n", b"\n"))
    return {"tool": "tools/fetch_models.py", "tool_sha256": text_sha(Path(__file__)),
            "procedural_tool": "tools/proc_furniture.py",
            "procedural_tool_sha256": text_sha(Path(__file__).parent / "proc_furniture.py"),
            "python": sys.version.split()[0], "packages": deps}


def write_manifest(out: Path, records: list[dict]) -> None:
    """MANIFEST.json: the converter, every converted model's source and licence, and every file
    in the folder (converted and procedural) with its SHA-256 and size."""
    files = {}
    for p in sorted(out.iterdir()):
        if p.name != "MANIFEST.json":
            data = p.read_bytes()
            files[p.name] = {"sha256": _sha256(data), "bytes": len(data)}
    converted = [r for r in records if "source_files_sha256" in r]
    procedural = sorted(p.stem for p in out.glob("proc_*.json"))
    manifest = {"converter": tool_record(),
                "models": [{k: r[k] for k in ("id", "name", "authors", "license", "source", "file",
                                               "source_files_sha256", "scale")} for r in converted],
                "procedural": [{"id": i, "source": "tools/proc_furniture.py"} for i in procedural],
                "files": files}
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8", newline="\n")


def build(out: Path, wanted: list[str]) -> list[dict]:
    """Convert the wanted models into a fresh folder, then replace the converted files in `out`
    (the procedural meshes there are kept). Nothing in `out` changes if any model fails a gate."""
    out.mkdir(parents=True, exist_ok=True)
    records = []
    with tempfile.TemporaryDirectory() as tmp:
        for model_id in wanted:
            size, cap, *caps = MODELS[model_id]
            rec = convert(model_id, size, cap, Path(tmp), caps[0] if caps else None)
            records.append(rec)
            c = rec["checks"]
            print(f"{model_id:26s} {len(rec['parts'])} parts {c['faces']:6d}/{c['source_faces']:6d} faces  "
                  f"size {rec['size_m']}  dev {c['max_vertex_distance_m'] * 1000:.1f} mm  "
                  f"cover {c['coverage_max_m'] * 1000:.1f}/{c['coverage_p999_m'] * 1000:.1f} mm")
        for p in out.iterdir():  # only the selected models are kept (procedural meshes are not ours)
            if not p.name.startswith("proc_"):
                p.unlink()
        for p in Path(tmp).iterdir():
            shutil.copy2(p, out / p.name)
    write_manifest(out, records)
    return records


def main() -> int:
    args = sys.argv[1:]
    if args == ["--check"]:
        with tempfile.TemporaryDirectory() as tmp:
            import proc_furniture  # the procedural meshes are regenerated too, then compared
            proc_furniture.OUT = Path(tmp)
            proc_furniture.main()
            build(Path(tmp), list(MODELS))
            a = {p.name: p.read_bytes() for p in Path(tmp).iterdir()}
            b = {p.name: p.read_bytes() for p in OUT.iterdir()}
            diff = sorted(n for n in a.keys() | b.keys() if a.get(n) != b.get(n))
            print("byte-identical" if not diff else f"DIFFERENT: {diff}")
            return 1 if diff else 0
    unknown = [a for a in args if a not in MODELS]
    if unknown:
        print(f"not in MODELS: {unknown}")
        return 2
    if args:  # convert these into the existing set
        with tempfile.TemporaryDirectory() as tmp:
            build(Path(tmp), args)
            for p in Path(tmp).iterdir():
                if p.name != "MANIFEST.json":
                    shutil.copy2(p, OUT / p.name)
        write_manifest(OUT, [json.loads(p.read_text(encoding="utf-8")) for p in sorted(OUT.glob("*.json"))
                             if p.name != "MANIFEST.json"])
        return 0
    build(OUT, list(MODELS))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
