"""Geometry helpers for the furniture models (development tools and tests): load a converted
model as one triangle mesh, and cut it at a height into a filled occupancy raster.

The raster is exact up to its cell size: every triangle crossing the plane adds its cut segment
(rasterized densely), and closed outlines are filled (scipy's binary_fill_holes), so a solid
seen in section counts as occupied inside, not only on its outline.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FURN = ROOT / "robot_env" / "assets" / "furniture"


def read_msh(data: bytes):
    nv, nn, nt, nf = np.frombuffer(data[:16], dtype="<i4")
    o = 16
    out = []
    for count, width, dtype in ((nv, 3, "<f4"), (nn, 3, "<f4"), (nt, 2, "<f4"), (nf, 3, "<i4")):
        size = int(count) * width * 4
        out.append(np.frombuffer(data[o:o + size], dtype=dtype).reshape(-1, width))
        o += size
    return tuple(out)


def model_mesh(model_id: str, folder: Path = FURN) -> tuple[np.ndarray, np.ndarray]:
    """(vertices, faces) of every part of a converted model, in the model's own frame (metres,
    base at z = 0, centred on x/y)."""
    rec = json.loads((folder / f"{model_id}.json").read_text(encoding="utf-8"))
    verts, faces, off = [], [], 0
    for p in rec["parts"]:
        v, _, _, f = read_msh((folder / p["mesh"]).read_bytes())
        verts.append(v.astype(float))
        faces.append(f.astype(np.int64) + off)
        off += len(v)
    return np.vstack(verts), np.vstack(faces)


def place(v: np.ndarray, pos, yaw_deg: float, scale: float = 1.0) -> np.ndarray:
    """Model-frame vertices turned by yaw (degrees, about z), scaled, then moved to pos."""
    c, s = np.cos(np.radians(yaw_deg)), np.sin(np.radians(yaw_deg))
    r = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return (v * scale) @ r.T + np.asarray(pos, float)


def cut_segments(v: np.ndarray, f: np.ndarray, z: float) -> np.ndarray:
    """(n, 2, 2) segments where the triangles cross the plane at height z."""
    t = v[f]  # (m, 3, 3)
    d = t[:, :, 2] - z
    d = np.where(d == 0.0, 1e-12, d)  # a vertex exactly on the plane counts as above it
    above = d > 0
    cross = above.any(1) & (~above).any(1)
    t, d = t[cross], d[cross]
    a_idx = np.array([0, 1, 2])
    b_idx = np.array([1, 2, 0])
    da, db = d[:, a_idx], d[:, b_idx]  # (m, 3) per edge
    hit = (da > 0) != (db > 0)  # exactly two edges per crossing triangle
    w = da / np.where(hit, da - db, 1.0)
    p = t[:, a_idx, :2] + (t[:, b_idx, :2] - t[:, a_idx, :2]) * w[:, :, None]  # (m, 3, 2)
    order = np.argsort(~hit, axis=1, kind="stable")[:, :2]  # the two hit edges
    return np.take_along_axis(p, order[:, :, None], axis=1)


def slice_raster(v: np.ndarray, f: np.ndarray, z: float, lo, hi, res: float = 0.005) -> np.ndarray:
    """Occupancy (bool, [ix, iy]) of the mesh's section at height z over the box lo..hi (x, y):
    cut outlines rasterized (sampled every half cell), closed outlines filled."""
    from scipy.ndimage import binary_fill_holes
    lo = np.asarray(lo, float)
    nx, ny = (np.ceil((np.asarray(hi, float) - lo) / res)).astype(int) + 1
    grid = np.zeros((nx, ny), bool)
    segs = cut_segments(v, f, z)
    if len(segs):
        a, b = segs[:, 0], segs[:, 1]
        n = (np.ceil(np.linalg.norm(b - a, axis=1) / (res / 2))).astype(int) + 1
        seg = np.repeat(np.arange(len(segs)), n)
        start = np.repeat(np.cumsum(n) - n, n)
        t = (np.arange(n.sum()) - start) / np.maximum(np.repeat(n, n) - 1, 1)
        pts = a[seg] + (b[seg] - a[seg]) * t[:, None]
        ij = np.floor((pts - lo) / res).astype(int)
        ok = (ij[:, 0] >= 0) & (ij[:, 0] < nx) & (ij[:, 1] >= 0) & (ij[:, 1] < ny)
        grid[ij[ok, 0], ij[ok, 1]] = True
    return binary_fill_holes(grid)


def describe(model_id: str, heights) -> None:
    """Print the occupied x/y extents of each connected piece of the section at each height."""
    from scipy.ndimage import label, find_objects
    v, f = model_mesh(model_id)
    lo, hi = v[:, :2].min(0) - 0.01, v[:, :2].max(0) + 0.01
    print(model_id, "bounds", v.min(0).round(3).tolist(), v.max(0).round(3).tolist())
    for z in heights:
        g = slice_raster(v, f, z, lo, hi)
        lab, n = label(g)
        parts = []
        for sl in find_objects(lab):
            x0, x1 = lo[0] + sl[0].start * 0.005, lo[0] + sl[0].stop * 0.005
            y0, y1 = lo[1] + sl[1].start * 0.005, lo[1] + sl[1].stop * 0.005
            if (x1 - x0) * (y1 - y0) > 1e-4:
                parts.append(f"x[{x0:.3f},{x1:.3f}] y[{y0:.3f},{y1:.3f}]")
        print(f"  z={z:.3f}: {n} pieces, filled {g.sum() * 25e-6:.3f} m2: " + "; ".join(parts[:6])
              + (f" (+{len(parts) - 6})" if len(parts) > 6 else ""))


if __name__ == "__main__":
    import sys
    hs = [0.005, 0.02, 0.05, 0.08, 0.1, 0.12, 0.14, 0.16, 0.2, 0.3, 0.4, 0.5, 0.7]
    for mid in sys.argv[1:]:
        describe(mid, hs)
