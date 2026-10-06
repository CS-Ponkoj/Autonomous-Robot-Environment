"""Procedural furniture meshes for objects without a fitting CC0 model (office chair, filing
cabinet, waste bin, floor lamp), written like converted models: <id>.json plus one binary .msh per
material in robot_env/assets/furniture/. Deterministic (pure arithmetic, no randomness), smooth
normals on curved parts, base at z = 0, origin at the base centre (the chair's column), front facing -y.

    .venv\\Scripts\\python tools\\proc_furniture.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from fetch_models import msh_bytes  # noqa: E402

OUT = ROOT / "robot_env" / "assets" / "furniture"

# material name -> (rgba, specular, shininess, reflectance)
MATERIALS = {
    "fabric": ((0.29, 0.3, 0.32, 1.0), 0.15, 0.22, 0.0),
    "piping": ((0.36, 0.37, 0.39, 1.0), 0.15, 0.22, 0.0),
    "plastic": ((0.11, 0.11, 0.12, 1.0), 0.35, 0.4, 0.0),
    "shell": ((0.4, 0.4, 0.41, 1.0), 0.3, 0.3, 0.0),
    "chrome": ((0.72, 0.73, 0.75, 1.0), 0.9, 0.9, 0.15),
    "brass": ((0.72, 0.56, 0.3, 1.0), 0.8, 0.7, 0.1),
    "steel_grey": ((0.64, 0.64, 0.61, 1.0), 0.5, 0.35, 0.02),
    "steel_dark": ((0.2, 0.21, 0.22, 1.0), 0.4, 0.5, 0.02),
    "bin_mesh": ((0.33, 0.34, 0.35, 1.0), 0.45, 0.5, 0.02),
    "liner": ((0.035, 0.035, 0.04, 1.0), 0.3, 0.35, 0.0),
    "shadow": ((0.03, 0.03, 0.03, 1.0), 0.0, 0.0, 0.0),
    "concrete": ((0.6, 0.59, 0.56, 1.0), 0.25, 0.3, 0.0),
    "label": ((1.0, 1.0, 1.0, 1.0), 0.1, 0.1, 0.0),
    "shade": ((0.96, 0.92, 0.82, 1.0), 0.05, 0.05, 0.0),
    "oak": ((1.0, 1.0, 1.0, 1.0), 0.2, 0.25, 0.0),
    "book_cloth": ((1.0, 1.0, 1.0, 1.0), 0.12, 0.15, 0.0),
    "book_cover": ((1.0, 1.0, 1.0, 1.0), 0.12, 0.15, 0.0),
    "upholstery": ((0.56, 0.52, 0.45, 1.0), 0.1, 0.2, 0.0),
    "piping_oat": ((0.48, 0.44, 0.38, 1.0), 0.1, 0.2, 0.0),
    "plinth": ((0.08, 0.07, 0.06, 1.0), 0.2, 0.3, 0.0),
}
OAK_RGB = (0.6, 0.43, 0.26)


# Procedural surface textures (seamless, seeded): material -> (tile size in metres, generator name).
# Parts without their own UVs get box-mapped UVs in metres / tile (each vertex projected on the plane
# facing its normal), so a texture keeps its real scale on every part; cylinders made with tile=
# carry their own wrapped UVs.
SURFACES = {
    "fabric": (0.08, "weave"), "shade": (0.12, "linen"), "steel_grey": (0.6, "powder"),
    "steel_dark": (0.3, "brushed"), "plastic": (0.25, "matte"), "shell": (0.25, "matte"), "bin_mesh": (0.06, "perforated"),
    "concrete": (0.3, "concrete"), "upholstery": (0.08, "weave"), "plinth": (0.25, "matte"),
}


def _periodic_noise(rng, size, beta, stretch=1.0):
    """Seamless noise: white noise filtered in the frequency domain (power ~ 1/f^beta), optionally
    stretched along x (streaks); normalized to zero mean and unit spread."""
    fx = np.fft.fftfreq(size)[None, :] / stretch
    fy = np.fft.fftfreq(size)[:, None]
    f = np.sqrt(fx * fx + fy * fy)
    f[0, 0] = 1.0
    spec = np.fft.fft2(rng.standard_normal((size, size))) / f ** (beta / 2)
    spec[0, 0] = 0.0
    n = np.real(np.fft.ifft2(spec))
    return (n - n.mean()) / (n.std() + 1e-12)


def surface_texture(kind: str, rgb, seed: int, size: int = 512):
    """A seamless texture image for a material colour rgb (0..1)."""
    from PIL import Image
    rng = np.random.default_rng(seed)
    base = np.asarray(rgb, float)[None, None, :3] * 255.0
    yy, xx = np.mgrid[0:size, 0:size] / size
    if kind == "weave":  # woven upholstery: a 2.5 mm over-under grid, heathered yarns, some fibre noise
        grid = 0.5 + 0.5 * np.sin(2 * np.pi * 32 * xx) * np.sin(2 * np.pi * 32 * yy)
        heather = _periodic_noise(rng, size, 0.4, 8.0) + _periodic_noise(rng, size, 0.4, 1 / 8.0)
        k = 1.0 + 0.16 * (grid - 0.5) + 0.05 * heather + 0.05 * _periodic_noise(rng, size, 1.0)
    elif kind == "linen":  # lampshade linen: fine weave, slubs, a little translucent variation
        weave = np.sin(2 * np.pi * 96 * xx) * 0.5 + np.sin(2 * np.pi * 96 * yy) * 0.5
        k = 1.0 + 0.035 * weave + 0.015 * _periodic_noise(rng, size, 2.5) + 0.015 * _periodic_noise(rng, size, 0.6, 6.0)
    elif kind == "powder":  # powder-coated steel: 1 to 2 mm speckle over gentle mottling
        k = 1.0 + 0.07 * _periodic_noise(rng, size, 0.2) + 0.015 * _periodic_noise(rng, size, 3.0)
    elif kind == "perforated":  # perforated steel sheet: 4 mm holes on a 6 mm grid (10 per tile), brushed
        d = np.hypot((xx * 10) % 1.0 - 0.5, (yy * 10) % 1.0 - 0.5) * 0.006  # metres from the hole centre
        hole = np.clip((0.0021 - d) / 0.0003, 0.0, 1.0)  # 4 mm across, anti-aliased edge
        streaks = 0.05 * _periodic_noise(rng, size, 1.2, 24.0) + 0.02 * _periodic_noise(rng, size, 0.3)
        k = (1.0 + streaks) * (1.0 - 0.85 * hole)
    elif kind == "concrete":  # cast concrete: fine aggregate speckle, mottling, a few pores
        pores = _periodic_noise(rng, size, 0.1)
        k = (1.0 + 0.06 * _periodic_noise(rng, size, 0.3) + 0.05 * _periodic_noise(rng, size, 2.5)
             - 0.25 * (pores > 2.6))
    elif kind == "brushed":  # brushed dark metal: long streaks
        k = 1.0 + 0.08 * _periodic_noise(rng, size, 1.2, 24.0) + 0.02 * _periodic_noise(rng, size, 0.3)
    elif kind == "oak":  # oak: long grain, growth rings, pores
        rings = np.sin(2 * np.pi * (6 * yy + 0.8 * _periodic_noise(rng, size, 3.0) * 0.15))
        k = 1.0 + 0.10 * rings + 0.08 * _periodic_noise(rng, size, 1.5, 12.0) + 0.03 * _periodic_noise(rng, size, 0.2)
    else:  # "matte": moulded plastic
        k = 1.0 + 0.03 * _periodic_noise(rng, size, 1.8) + 0.015 * _periodic_noise(rng, size, 0.2)
    img = np.clip(base * k[..., None], 0, 255).astype(np.uint8)
    return Image.fromarray(img)


def _box_uv(v, n, tile):
    """UVs in tiles: each vertex projected on the plane facing its normal's main axis."""
    a = np.abs(n)
    axis = np.argmax(a, axis=1)
    uv = np.empty((len(v), 2))
    uv[axis == 0] = v[axis == 0][:, [1, 2]]
    uv[axis == 1] = v[axis == 1][:, [0, 2]]
    uv[axis == 2] = v[axis == 2][:, [0, 1]]
    return uv / tile


class Mesh:
    """Triangles collected per material."""

    def __init__(self):
        self.parts: dict[str, list] = {}

    def add(self, material, v, f, n=None, uv=None):
        v = np.asarray(v, float)
        f = np.asarray(f, np.int64)
        if n is None:
            n = _flat_normals(v, f)
        uv = np.zeros((len(v), 2)) if uv is None else np.asarray(uv, float)
        self.parts.setdefault(material, []).append((v, f, np.asarray(n, float), uv))

    def textured_box(self, material, lo, hi, face_uv, face_mat=None, rot=None, pivot=(0.0, 0.0, 0.0)):
        """A box from corner lo to corner hi with its own texture rectangle per face: face_uv maps
        "-x", "+x", "-y", "+y", "-z", "+z" to (u0, v0, u1, v1) (v down the image; u0 > u1 mirrors);
        on the side faces v runs down from the top edge, u along the face as seen from outside.
        face_mat: a face -> another material (its own texture). rot: a 3x3 rotation about pivot."""
        (x0, y0, z0), (x1, y1, z1) = lo, hi
        face_mat = face_mat or {}
        piv = np.asarray(pivot, float)
        faces = {  # top-left, top-right, bottom-right, bottom-left, seen from outside
            "-y": ((x0, y0, z1), (x1, y0, z1), (x1, y0, z0), (x0, y0, z0)),
            "+y": ((x1, y1, z1), (x0, y1, z1), (x0, y1, z0), (x1, y1, z0)),
            "-x": ((x0, y1, z1), (x0, y0, z1), (x0, y0, z0), (x0, y1, z0)),
            "+x": ((x1, y0, z1), (x1, y1, z1), (x1, y1, z0), (x1, y0, z0)),
            "+z": ((x0, y1, z1), (x1, y1, z1), (x1, y0, z1), (x0, y0, z1)),
            "-z": ((x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0)),
        }
        for key, quad in faces.items():
            u0, v0, u1, v1 = face_uv[key]
            uv = np.array([(u0, v0), (u1, v0), (u1, v1), (u0, v1)], float)
            q = np.array(quad, float)
            if rot is not None:
                q = (q - piv) @ np.asarray(rot, float).T + piv
            self.add(face_mat.get(key, material), q, [(0, 2, 1), (0, 3, 2)], uv=uv)

    def rounded_box(self, material, center, half, r, pitch_deg=0.0, piping=(), pipe_r=0.004,
                    pipe_mat="piping", n_arc=3):
        """A box with every edge and corner rounded to radius r, smooth normals (a cushion, a
        powder-coated panel). piping: faces ("+z", "-y", ...) whose rim gets a bead of radius
        pipe_r (pipe_mat) on the rounded edge. Turned pitch_deg about x through its centre."""
        h = np.asarray(half, float)
        a = h - r
        def samples(k):
            band = a[k] + r * np.linspace(0.0, 1.0, n_arc + 1)[1:]
            return np.concatenate([-band[::-1], np.linspace(-a[k], a[k], 3), band])
        for ax in range(3):
            u, w = [k for k in range(3) if k != ax]
            su, sw = samples(u), samples(w)
            for s in (-1.0, 1.0):
                g = np.zeros((len(su), len(sw), 3))
                g[..., ax] = s * h[ax]
                g[..., u] = su[:, None]
                g[..., w] = sw[None, :]
                g = g.reshape(-1, 3)
                c = np.clip(g, -a, a)
                n = g - c
                n /= np.linalg.norm(n, axis=1)[:, None]
                v = c + r * n
                f = []
                nw = len(sw)
                for i in range(len(su) - 1):
                    for j in range(nw - 1):
                        p, q2, t, o = i * nw + j, (i + 1) * nw + j, (i + 1) * nw + j + 1, i * nw + j + 1
                        for tri in ((p, q2, t), (p, t, o)):
                            gn = np.cross(v[tri[1]] - v[tri[0]], v[tri[2]] - v[tri[0]])
                            f.append(tri if gn @ n[list(tri)].sum(0) >= 0 else (tri[0], tri[2], tri[1]))
                self.add(material, _turn(v, 0.0, pitch_deg) + np.asarray(center, float), f,
                         _turn(n, 0.0, pitch_deg))
        for face in piping:
            ax = "xyz".index(face[1])
            s = 1.0 if face[0] == "+" else -1.0
            u, w = [k for k in range(3) if k != ax]
            k = 1 / math.sqrt(2)
            path, out = [], []
            for cu, cw, a0 in ((1, 1, 0.0), (-1, 1, 90.0), (-1, -1, 180.0), (1, -1, 270.0)):
                for t in np.radians(a0 + np.linspace(0.0, 90.0, 6)):
                    d = np.zeros(3)
                    d[u], d[w] = math.cos(t), math.sin(t)
                    p = np.zeros(3)
                    p[u], p[w] = cu * a[u], cw * a[w]
                    p[ax] = s * a[ax]
                    nrm = k * d
                    nrm[ax] = s * k
                    path.append(p + r * nrm)
                    out.append(nrm)
            self.tube(pipe_mat, np.array(path), np.array(out), pipe_r, center, pitch_deg)

    def tube(self, material, path, out, radius, center, pitch_deg=0.0, seg=8):
        """A closed tube of `radius` along `path` (rows: points), its ring frame set by `out` (a
        unit vector per point, normal to the path); smooth normals."""
        m = len(path)
        tang = np.roll(path, -1, 0) - np.roll(path, 1, 0)
        tang /= np.maximum(np.linalg.norm(tang, axis=1), 1e-12)[:, None]
        e1 = out - (out * tang).sum(1)[:, None] * tang
        e1 /= np.linalg.norm(e1, axis=1)[:, None]
        e2 = np.cross(tang, e1)
        ang = 2 * np.pi * np.arange(seg) / seg
        n = (np.cos(ang)[None, :, None] * e1[:, None, :] + np.sin(ang)[None, :, None] * e2[:, None, :]).reshape(-1, 3)
        v = np.repeat(path, seg, 0) + radius * n
        f = []
        for i in range(m):
            i2 = (i + 1) % m
            for j in range(seg):
                j2 = (j + 1) % seg
                a_, b_, c_, d_ = i * seg + j, i2 * seg + j, i2 * seg + j2, i * seg + j2
                for tri in ((a_, b_, c_), (a_, c_, d_)):
                    gn = np.cross(v[tri[1]] - v[tri[0]], v[tri[2]] - v[tri[0]])
                    f.append(tri if gn @ n[list(tri)].sum(0) >= 0 else (tri[0], tri[2], tri[1]))
        self.add(material, _turn(v, 0.0, pitch_deg) + np.asarray(center, float), f, _turn(n, 0.0, pitch_deg))

    def annulus(self, material, z, r_in, r_out, seg=32, up=True):
        """A flat ring at height z between r_in and r_out, facing up (or down)."""
        a = 2 * math.pi * np.arange(seg) / seg
        ring = np.stack([np.cos(a), np.sin(a)], 1)
        v = np.vstack([np.column_stack([ring * r_in, np.full(seg, z)]), np.column_stack([ring * r_out, np.full(seg, z)])])
        f = []
        for i in range(seg):
            j = (i + 1) % seg
            f += [(i, seg + i, seg + j), (i, seg + j, j)] if up else [(i, seg + j, seg + i), (i, j, seg + j)]
        self.add(material, v, f, np.tile([0.0, 0.0, 1.0 if up else -1.0], (2 * seg, 1)))

    def box(self, material, center, half, yaw_deg=0.0, pitch_deg=0.0):
        """A box (flat-shaded): 24 vertices, 12 triangles; turned pitch about x, then yaw about z."""
        hx, hy, hz = half
        corners = np.array([[sx * hx, sy * hy, sz * hz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
        faces_q = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
        v, f = [], []
        for q in faces_q:
            base = len(v)
            v.extend(corners[list(q)])
            f.extend([(base, base + 1, base + 2), (base, base + 2, base + 3)])
        v = _turn(np.array(v), yaw_deg, pitch_deg) + np.asarray(center, float)
        self.add(material, v, f)

    def cylinder(self, material, center, radius, half_h, axis="z", seg=16, caps=True, radius_top=None,
                 tile=None, inward=False):
        """A (possibly tapered) cylinder with smooth sides. tile: wrap UVs around it (a whole number
        of tiles of about `tile` metres round, v = height / tile), for an upright cylinder.
        inward: the side faces in (the inside of a liner)."""
        rt = radius if radius_top is None else radius_top
        if tile is not None:
            self._wrapped_cylinder(material, center, radius, rt, half_h, seg, caps, tile, inward)
            return
        a = 2 * math.pi * np.arange(seg) / seg
        ring = np.stack([np.cos(a), np.sin(a)], 1)
        bot = np.column_stack([ring * radius, np.full(seg, -half_h)])
        top = np.column_stack([ring * rt, np.full(seg, half_h)])
        slope = (radius - rt) / (2 * half_h)
        side_n = np.column_stack([ring, np.full(seg, slope)])
        side_n /= np.linalg.norm(side_n, axis=1)[:, None]
        v = [bot, top]
        n = [side_n, side_n]
        f = []
        for i in range(seg):
            j = (i + 1) % seg
            f += [(i, j, seg + j), (i, seg + j, seg + i)]
        if caps:
            for z, r, sign in ((-half_h, radius, -1), (half_h, rt, 1)):
                base = sum(len(x) for x in v)
                v.append(np.vstack([[0.0, 0.0, z], np.column_stack([ring * r, np.full(seg, z)])]))
                n.append(np.tile([0.0, 0.0, sign], (seg + 1, 1)))
                for i in range(seg):
                    j = (i + 1) % seg
                    f.append((base, base + 1 + i, base + 1 + j) if sign > 0 else (base, base + 1 + j, base + 1 + i))
        v, n = np.vstack(v), np.vstack(n)
        if axis == "x":
            rot = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], float)
            v, n = v @ rot.T, n @ rot.T
        elif axis == "y":
            rot = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], float)
            v, n = v @ rot.T, n @ rot.T
        self.add(material, v + np.asarray(center, float), f, n)

    def _wrapped_cylinder(self, material, center, rb, rt, half_h, seg, caps, tile, inward):
        reps = max(1, round(math.pi * (rb + rt) / tile))
        a = 2 * math.pi * np.arange(seg + 1) / seg  # the seam column is doubled so u runs 0..reps
        ring = np.stack([np.cos(a), np.sin(a)], 1)
        slope = (rb - rt) / (2 * half_h)
        sn = np.column_stack([ring, np.full(seg + 1, slope)])
        sn /= np.linalg.norm(sn, axis=1)[:, None]
        sign = -1.0 if inward else 1.0
        z0, z1 = center[2] - half_h, center[2] + half_h
        v = np.vstack([np.column_stack([ring * rb, np.full(seg + 1, -half_h)]),
                       np.column_stack([ring * rt, np.full(seg + 1, half_h)])])
        uv = np.vstack([np.column_stack([a / (2 * math.pi) * reps, np.full(seg + 1, -z0 / tile)]),
                        np.column_stack([a / (2 * math.pi) * reps, np.full(seg + 1, -z1 / tile)])])
        f = []
        s1 = seg + 1
        for i in range(seg):
            tris = [(i, i + 1, s1 + i + 1), (i, s1 + i + 1, s1 + i)]
            f += tris if not inward else [(t[0], t[2], t[1]) for t in tris]
        self.add(material, v + np.asarray(center, float), f, sign * np.vstack([sn, sn]), uv=uv)
        if caps:
            for z, r, up in ((-half_h, rb, False), (half_h, rt, True)):
                c = np.asarray(center, float) + [0.0, 0.0, z]
                pts = np.vstack([[0.0, 0.0], ring[:-1] * r])
                vv = np.column_stack([pts, np.zeros(len(pts))]) + c
                ff = [(0, 1 + i, 1 + (i + 1) % seg) if up else (0, 1 + (i + 1) % seg, 1 + i) for i in range(seg)]
                self.add(material, vv, ff, np.tile([0.0, 0.0, 1.0 if up else -1.0], (len(vv), 1)),
                         uv=vv[:, :2] / tile)

    def cushion(self, material, center, half, e=0.25, nu=20, nv=12, pitch_deg=0.0):
        """A rounded block (superellipsoid, exponent e: small = boxier) with smooth normals."""
        hx, hy, hz = half
        u = np.linspace(-math.pi, math.pi, nu, endpoint=False)
        w = np.linspace(-math.pi / 2, math.pi / 2, nv)
        uu, ww = np.meshgrid(u, w, indexing="ij")

        def sp(x, p):
            return np.sign(x) * np.abs(x) ** p

        x = hx * sp(np.cos(ww), e) * sp(np.cos(uu), e)
        y = hy * sp(np.cos(ww), e) * sp(np.sin(uu), e)
        z = hz * sp(np.sin(ww), e)
        nx = sp(np.cos(ww), 2 - e) * sp(np.cos(uu), 2 - e) / hx
        ny = sp(np.cos(ww), 2 - e) * sp(np.sin(uu), 2 - e) / hy
        nz = sp(np.sin(ww), 2 - e) / hz
        v = np.stack([x, y, z], -1).reshape(-1, 3)
        n = np.stack([nx, ny, nz], -1).reshape(-1, 3)
        n /= np.maximum(np.linalg.norm(n, axis=1), 1e-12)[:, None]
        poles = np.linalg.norm(n, axis=1) < 0.5
        n[poles] = np.array([0.0, 0.0, 1.0]) * np.sign(v[poles, 2:3] + 1e-12)
        f = []
        for i in range(nu):
            i2 = (i + 1) % nu
            for j in range(nv - 1):
                a, b, c, d = i * nv + j, i2 * nv + j, i2 * nv + j + 1, i * nv + j + 1
                f += [(a, b, c), (a, c, d)]
        v = _turn(v, 0.0, pitch_deg) + np.asarray(center, float)
        n = _turn(n, 0.0, pitch_deg)
        self.add(material, v, f, n)

    def write(self, model_id: str, name: str, members_note: str, textures: dict | None = None,
              base_at_zero: bool = True) -> dict:
        """textures: material name -> PIL image, written as <part>.png and used by that part.
        base_at_zero: move the mesh so its lowest point is at z = 0 (furniture standing on the
        floor); False keeps the builder's heights (books on a shelf's boards)."""
        parts = []
        for old in OUT.glob(f"{model_id}_*"):  # this model's parts from an earlier build
            if old.suffix in (".msh", ".png") and old.stem[len(model_id) + 1:].isdigit():
                old.unlink()
        allv = np.vstack([v for chunks in self.parts.values() for v, _, _, _ in chunks])
        lo, hi = allv.min(0), allv.max(0)
        shift = np.array([0.0, 0.0, -lo[2] if base_at_zero else 0.0])  # kept in the builder's frame
        for k, (mat, chunks) in enumerate(sorted(self.parts.items())):
            surface = SURFACES.get(mat) if not (textures and mat in textures) else None
            vs, fs, ns, uvs, off = [], [], [], [], 0
            for v, f, n, uv in chunks:
                if surface is not None and not np.any(uv):  # a part without its own UVs
                    uv = _box_uv(v + shift, n, surface[0])
                vs.append(v + shift)
                fs.append(f + off)
                ns.append(n)
                uvs.append(uv)
                off += len(v)
            v, f, n, uv = np.vstack(vs), np.vstack(fs), np.vstack(ns), np.vstack(uvs)
            mesh = f"{model_id}_{k}.msh"
            (OUT / mesh).write_bytes(msh_bytes(v, n, uv, f))
            rgba, spec, shin, refl = MATERIALS[mat]
            texture = None
            image = textures.get(mat) if textures else None
            if isinstance(image, str):  # a texture file shared by several models (written once)
                texture, image, rgba = image, None, (1.0, 1.0, 1.0, 1.0)
            if image is None and surface is not None:
                seed = int.from_bytes(f"{model_id}/{mat}".encode()[:8].ljust(8, b"0"), "little") % (2 ** 31)
                image = surface_texture(surface[1], rgba, seed)
            if image is not None:
                import io
                texture = f"{model_id}_{k}.png"
                buf = io.BytesIO()
                image.save(buf, "PNG", optimize=True)
                (OUT / texture).write_bytes(buf.getvalue())
                rgba = (1.0, 1.0, 1.0, 1.0)
            parts.append({"mesh": mesh, "material": mat, "texture": texture, "rgba": list(rgba), "specular": spec,
                          "shininess": shin, "reflectance": refl, "faces": int(len(f)), "vertices": int(len(v))})
        rec = {"id": model_id, "name": name, "authors": [], "license": "procedural (this repository)",
               "source": "tools/proc_furniture.py", "size_m": [round(float(x), 4) for x in hi - lo],
               "parts": parts, "note": members_note}
        (OUT / f"{model_id}.json").write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8", newline="\n")
        return rec


def _flat_normals(v, f):
    n = np.zeros_like(v)
    fn = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    for c in range(3):
        np.add.at(n, f[:, c], fn)
    return n / np.maximum(np.linalg.norm(n, axis=1), 1e-30)[:, None]


def _turn(v, yaw_deg, pitch_deg):
    p = math.radians(pitch_deg)
    rx = np.array([[1, 0, 0], [0, math.cos(p), -math.sin(p)], [0, math.sin(p), math.cos(p)]])
    y = math.radians(yaw_deg)
    rz = np.array([[math.cos(y), -math.sin(y), 0], [math.sin(y), math.cos(y), 0], [0, 0, 1]])
    return v @ (rz @ rx).T


# ---- the objects (dimensions in metres; each function documents the members that fit it) ----

# Every part that a robot could touch (from the floor to its top, 0.1355 m) crosses the lidar
# plane or stands under one that does (tools/furniture_check.py lidar gate): no star bases, no low
# stretchers, no thin low discs, until the robot has height-aware sensing.
CHAIR = {"leg": 0.0125, "leg_x": 0.205, "leg_y": 0.195, "pan_top": 0.395, "seat_top": 0.48,
         "back_top": 0.95}


def office_chair() -> dict:
    """Four-legged office chair: square steel legs (2.5 cm) from the floor to the seat, the rear
    legs going on up as the back posts; a 1 cm moulded seat pan under a piped seat cushion
    0.48 x 0.46 m (1 cm overhang), top at 0.48 m; a piped back pad reclined 8 degrees on a moulded
    polypropylene shell, held to the posts by two tabs a side, to 0.95 m; armrests with moulded pads. Front
    faces -y."""
    m = Mesh()
    c = CHAIR
    lg = c["leg"]
    for sx in (-1, 1):
        x = sx * c["leg_x"]
        m.box("chrome", (x, -c["leg_y"], c["pan_top"] / 2), (lg, lg, c["pan_top"] / 2))  # front leg
        m.box("chrome", (x, c["leg_y"], c["back_top"] / 2), (lg, lg, c["back_top"] / 2))  # rear leg and back post
        m.cylinder("plastic", (x, -c["leg_y"], 0.004), lg + 0.002, 0.004, seg=12)  # floor glides
        m.cylinder("plastic", (x, c["leg_y"], 0.004), lg + 0.002, 0.004, seg=12)
        m.box("chrome", (x, 0.0, 0.62), (lg * 0.8, c["leg_y"], lg * 0.8))  # arm rail
        m.box("chrome", (x, -c["leg_y"] + 0.02, 0.51), (lg * 0.8, lg * 0.8, 0.11))  # arm front post
        m.rounded_box("plastic", (x, -0.02, 0.645), (0.028, 0.13, 0.013), 0.011)  # moulded arm pad
        for zb, yb in BACK_BRACKETS:  # slim tabs flush with the post's inner face, into the shell's side
            m.box("chrome", (sx * 0.19025, (0.195 + yb + 0.006) / 2, zb), (0.00225, (yb + 0.006 - 0.195) / 2, 0.012))
    m.rounded_box("plastic", (0, 0, 0.385), (0.23, 0.22, 0.005), 0.004)  # seat pan, its lip under the seat
    m.rounded_box("fabric", (0, 0, 0.435), (0.24, 0.23, 0.045), 0.016, piping=("+z",))
    m.rounded_box("fabric", BACK_CENTRE, BACK_HALF, 0.016, pitch_deg=-BACK_RECLINE, piping=("-y",))
    tilt = math.radians(BACK_RECLINE)
    off = BACK_HALF[1] - 0.002  # the moulded polypropylene shell: a tray the pad sits in, its lip round it
    m.rounded_box("shell", (0.0, BACK_CENTRE[1] + off * math.cos(tilt), BACK_CENTRE[2] - off * math.sin(tilt)),
                  (BACK_HALF[0] + 0.003, 0.01, BACK_HALF[2] + 0.003), 0.006, pitch_deg=-BACK_RECLINE)
    return m.write("proc_office_chair", "Office chair (procedural)",
                   "members: four legs, two back posts, glides, seat pan, seat, back, arm rails, arm posts, pads")


BACK_CENTRE, BACK_HALF, BACK_RECLINE = (0.0, 0.19, 0.75), (0.185, 0.03, 0.17), 8.0
# the tabs (height, y of the middle of the shell's side there) at z 0.68 and 0.82 m
BACK_BRACKETS = tuple((zb, round(BACK_CENTRE[1] + (BACK_HALF[1] - 0.002) / math.cos(math.radians(BACK_RECLINE))
                                 + (zb - BACK_CENTRE[2]) * math.tan(math.radians(BACK_RECLINE)), 4))
                      for zb in (0.68, 0.82))


def label_atlas():
    """Four drawer labels stacked (128 x 48 px each): a steel frame, a paper card, one written line."""
    from PIL import Image
    rng = np.random.default_rng(7)
    img = np.zeros((192, 128, 3), float)
    for k in range(4):
        card = np.full((48, 128, 3), 236.0) * (1.0 + 0.02 * rng.standard_normal((48, 128, 1)))
        card[:5], card[-5:], card[:, :5], card[:, -5:] = 150.0, 150.0, 150.0, 150.0  # holder frame
        x = 16
        for _ in range(int(rng.integers(2, 4))):  # a hand-written title: a few strokes
            w = int(rng.integers(14, 34))
            card[20:27, x:x + w] = (40.0, 44.0, 70.0)
            x += w + int(rng.integers(5, 9))
            if x > 104:
                break
        img[48 * k:48 * (k + 1)] = card
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))


def filing_cabinet() -> dict:
    """Four-drawer steel filing cabinet, 0.47 wide, 0.62 deep, 1.32 m tall, powder-coated: edges
    rounded 3 mm, a 1 cm top with a 4 mm lip, drawer fronts 6 mm proud over dark gaps, a pull
    handle and a labelled holder on each, a lock on the top drawer."""
    m = Mesh()
    w, d, h = 0.235, 0.31, 1.32
    m.rounded_box("steel_grey", (0, 0.0, (h - 0.01) / 2), (w, d - 0.006, (h - 0.01) / 2), 0.003)  # body
    m.rounded_box("steel_grey", (0, -0.003, h - 0.005), (w + 0.004, d - 0.003, 0.005), 0.003)  # top and lip
    m.box("steel_dark", (0, -0.006, 0.02), (w - 0.01, d - 0.012, 0.02))  # recessed plinth
    gap = 0.006
    top = h - 0.03
    bottom = 0.05
    m.box("shadow", (0, -d + 0.0055, (top + bottom) / 2), (w - 0.006, 0.0005, (top - bottom) / 2))  # the gaps
    dh = (top - bottom) / 4
    for k in range(4):
        z0 = bottom + k * dh + gap / 2
        z1 = bottom + (k + 1) * dh - gap / 2
        zc = (z0 + z1) / 2
        m.rounded_box("steel_grey", (0, -d + 0.003, zc), (w - 0.012, 0.003, (z1 - z0) / 2), 0.0025)
        m.box("chrome", (0, -d - 0.012, zc + 0.06), (0.06, 0.006, 0.008))  # handle bar
        for s in (-1, 1):
            m.box("chrome", (s * 0.055, -d - 0.006, zc + 0.06), (0.005, 0.006, 0.008))
        lab = ((0.002, (k * 48 + 1) / 192, 0.998, ((k + 1) * 48 - 1) / 192))
        edge = (0.01, (k * 48 + 1) / 192, 0.02, (k * 48 + 3) / 192)
        m.textured_box("label", (-0.04, -d - 0.002, zc + 0.095), (0.04, -d, zc + 0.125),
                       {"-y": lab, "+y": edge, "-x": edge, "+x": edge, "-z": edge, "+z": edge})
    m.cylinder("chrome", (0.17, -d - 0.002, top - 0.035), 0.009, 0.002, axis="y", seg=20)  # lock
    m.box("shadow", (0.17, -d - 0.0042, top - 0.035), (0.0012, 0.0003, 0.005))  # its keyway
    return m.write("proc_filing_cabinet", "Filing cabinet (procedural)",
                   "members: body box, a handle box per drawer", {"label": label_atlas()})


def waste_bin() -> dict:
    """Round perforated-steel waste bin: 0.29 m diameter at the rim, 0.25 m at the base, 0.34 m
    tall, rolled rim; a black liner inside, its rolled lip 1 cm above the rim."""
    m = Mesh()
    m.cylinder("bin_mesh", (0, 0, 0.17), 0.125, 0.17, seg=40, caps=False, radius_top=0.145, tile=0.06)
    m.cylinder("steel_dark", (0, 0, 0.004), 0.125, 0.004, seg=40)
    a = 2 * math.pi * np.arange(40) / 40  # the rolled rim: a closed tube round the top
    ring = np.column_stack([np.cos(a), np.sin(a), np.zeros(40)])
    m.tube("chrome", ring * 0.146, ring, 0.005, (0.0, 0.0, 0.337))
    m.cylinder("liner", (0, 0, 0.1885), 0.12, 0.1635, seg=40, caps=False, radius_top=0.14)  # liner
    m.cylinder("liner", (0, 0, 0.1885), 0.119, 0.1635, seg=40, caps=False, radius_top=0.138, tile=0.3,
               inward=True)  # its inside
    m.annulus("liner", 0.026, 0.001, 0.119, seg=40)  # the liner's bottom, seen from above
    m.cylinder("liner", (0, 0, 0.348), 0.142, 0.004, seg=40, caps=False)  # its rolled lip, 1 cm above the rim
    m.annulus("liner", 0.352, 0.138, 0.142, seg=40)
    return m.write("proc_waste_bin", "Waste bin (procedural)", "members: tapered stack of cylinders")


def floor_lamp() -> dict:
    """Floor lamp on a cast concrete base (0.26 m across, 0.16 m tall with a 1 cm chamfer: it
    crosses the lidar plane), a brass collar, a slim pole, and a fabric drum shade at 1.45 to 1.65 m."""
    m = Mesh()
    m.cylinder("concrete", (0, 0, 0.075), 0.13, 0.075, seg=48, tile=0.3)
    m.cylinder("concrete", (0, 0, 0.155), 0.13, 0.005, seg=48, radius_top=0.12, tile=0.3)  # chamfer
    m.cylinder("brass", (0, 0, 0.175), 0.024, 0.015, seg=24)  # collar
    m.cylinder("brass", (0, 0, 0.191), 0.02, 0.001, seg=24, radius_top=0.016)
    m.cylinder("chrome", (0, 0, 0.82), 0.012, 0.635, seg=12)
    m.cylinder("shade", (0, 0, 1.55), 0.2, 0.1, seg=32, caps=False, radius_top=0.17)
    return m.write("proc_floor_lamp", "Floor lamp (procedural)", "members: base, collar, pole, shade")


OAK_PX = 512  # pixels per 0.5 m (about 1 mm a pixel)
OAK_REGIONS = {  # name: (column, row) of a 512 x 512 px (0.5 x 0.5 m) region in the 1536 x 1024 atlas
    "flat_a": (0, 0), "flat_b": (0, 1), "quarter_a": (1, 0), "quarter_b": (1, 1), "end": (2, 0), "edge": (2, 1)}


def oak_atlas(seed: int = 5):
    """Oak at real scale, 0.5 x 0.5 m regions with the grain along u: two flat-sawn faces
    (cathedral arches from rings cut at a slant, ring pitch drifting 4.5 to 11 mm), two
    quarter-sawn faces (straight close rings with ray flecks), end grain (ring arcs) and a planed
    edge; each region its own tone (3 to 6 percent apart), pores along the grain."""
    from PIL import Image
    rng = np.random.default_rng(seed)
    base = np.array(OAK_RGB) * 255.0
    n = OAK_PX
    vv, uu = np.mgrid[0:n, 0:n] / n * 0.5  # metres across and along the grain

    def rings(r, pitch0):
        """Ring phase at radius r (metres from the pith): the pitch drifts with the growth years."""
        pitch = pitch0 * (1.0 + 0.4 * np.sin(r * 31.0 + rng.uniform(0, 6.3)) + 0.15 * np.sin(r * 97.0))
        return r / pitch

    def finish(phase, tone, extra=0.0):
        late = (0.5 + 0.5 * np.sin(2 * np.pi * phase)) ** 5
        pores = _periodic_noise(rng, n, 0.2, 10.0)
        k = (tone - 0.17 * late + 0.045 * _periodic_noise(rng, n, 1.5, 12.0) - 0.09 * (pores > 2.0) + extra)
        return base * k[..., None]

    img = np.zeros((2 * n, 3 * n, 3))

    def put(name, tile):
        c, r = OAK_REGIONS[name]
        img[r * n:(r + 1) * n, c * n:(c + 1) * n] = tile

    for name, tone, centre in (("flat_a", 1.0, 0.21), ("flat_b", 0.95, 0.31)):
        depth = 0.012 + 0.05 * uu / 0.5 + 0.004 * _periodic_noise(rng, n, 3.0, 4.0)  # the cut runs out of the log
        r = np.hypot(depth, vv - centre) + 0.06
        put(name, finish(rings(r, 0.0075), tone))
    for name, tone in (("quarter_a", 1.03), ("quarter_b", 0.97)):
        r = 0.06 + vv + 0.002 * _periodic_noise(rng, n, 3.0, 6.0)
        fleck = _periodic_noise(rng, n, 1.2, 5.0)
        put(name, finish(rings(r, 0.0055), tone, 0.12 * np.clip(fleck - 1.4, 0.0, 1.0)))
    yy, xx = np.mgrid[0:n, 0:n] / n * 0.5
    r = np.hypot(xx - 0.13, yy + 0.35) + 0.0015 * _periodic_noise(rng, n, 3.0)
    late = (0.5 + 0.5 * np.sin(2 * np.pi * rings(r, 0.0075))) ** 4
    put("end", base * (0.82 - 0.15 * late + 0.06 * _periodic_noise(rng, n, 0.2))[..., None])
    depth = 0.02 + 0.03 * uu / 0.5
    put("edge", finish(rings(np.hypot(depth, vv - 0.25) + 0.06, 0.0075), 1.1))
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))


def oak_texture() -> str:
    """The oak atlas as one file shared by every oak model (written on first use in a build)."""
    name = "proc_oak.png"
    if name not in _SHARED:
        import io
        buf = io.BytesIO()
        oak_atlas().save(buf, "PNG", optimize=True)
        (OUT / name).write_bytes(buf.getvalue())
        _SHARED.add(name)
    return name


_SHARED: set[str] = set()


def _oak(L, H, region, rnd):
    """A texture rectangle for an oak face L m along the grain and H m across it, from an atlas
    region (OAK_REGIONS), at a random place in it."""
    c, r = OAK_REGIONS[region]
    su, sv = OAK_PX / (3 * OAK_PX) / 0.5, OAK_PX / (2 * OAK_PX) / 0.5  # atlas units per metre
    L = min(L, 0.5)  # a face longer than a region takes all of it, its grain stretched along its length
    du, dv = rnd.uniform(0, 0.5 - L), rnd.uniform(0, 0.5 - H)
    u0, v0 = c / 3 + du * su, r / 2 + dv * sv
    return (u0, v0, u0 + L * su, v0 + H * sv)


def plant_stand() -> dict:
    """Oak plant stand, 0.44 x 0.44 m, 0.22 m tall: four mitred sides with the grain running round
    them (two flat-sawn, two quarter-sawn), a 2 cm flat-sawn top board with a 1 cm overhang (end
    grain on two edges), on a recessed dark plinth. A planter stands on it, so the planter's
    flared foot is above the robot."""
    import random
    rnd = random.Random(3)
    m = Mesh()
    b = 0.22
    m.textured_box("oak", (-b, -b, 0.03), (b, b, 0.2),
                   {"-y": _oak(2 * b, 0.17, "flat_b", rnd), "+y": _oak(2 * b, 0.17, "flat_a", rnd),
                    "-x": _oak(2 * b, 0.17, "quarter_a", rnd), "+x": _oak(2 * b, 0.17, "quarter_b", rnd),
                    "+z": _oak(2 * b, 2 * b, "flat_b", rnd), "-z": _oak(2 * b, 2 * b, "flat_b", rnd)})
    t = 0.23
    m.textured_box("oak", (-t, -t, 0.2), (t, t, 0.22),
                   {"+z": _oak(2 * t, 2 * t, "flat_a", rnd), "-z": _oak(2 * t, 2 * t, "flat_a", rnd),
                    "-y": _oak(2 * t, 0.02, "edge", rnd), "+y": _oak(2 * t, 0.02, "edge", rnd),
                    "-x": _oak(2 * t, 0.02, "end", rnd), "+x": _oak(2 * t, 0.02, "end", rnd)})
    for sx in (-1, 1):  # the mitre joints: a fine dark line down each corner
        for sy in (-1, 1):
            m.box("shadow", (sx * b, sy * b, 0.115), (0.0007, 0.0007, 0.085), yaw_deg=45.0)
    m.box("steel_dark", (0, 0, 0.015), (0.205, 0.205, 0.015))  # plinth, set back 1.5 cm
    return m.write("proc_plant_stand", "Plant stand (procedural)", "members: body, top board, plinth",
                   {"oak": oak_texture()})


def shelf_feet() -> dict:
    """Four levelling feet (3.3 cm: a pad and a threaded stud) under a steel_frame_shelves_01 post
    pattern (posts at x +-0.534,
    y +-0.236 in the shelf's frame); the shelf stands on them, its bottom level crossing the lidar
    plane."""
    m = Mesh()
    for sx in (-1, 1):
        for sy in (-1, 1):
            m.cylinder("plastic", (sx * 0.534, sy * 0.236, 0.006), 0.017, 0.006, seg=16)
            m.cylinder("chrome", (sx * 0.534, sy * 0.236, 0.0225), 0.008, 0.0105, seg=12)
    return m.write("proc_shelf_feet", "Shelf levelling feet (procedural)", "members: four foot cylinders")


BOOK_COLOURS = ((110, 30, 32), (30, 42, 72), (36, 62, 42), (168, 128, 52), (214, 200, 166), (26, 26, 30),
                (92, 62, 40), (112, 112, 118), (40, 80, 86), (82, 26, 42), (150, 70, 40), (60, 72, 110))
ATLAS_W, ATLAS_H, COL = 1024, 256, 16  # 63 spine designs in 16 px columns, the last column is page edges
COVER_W, COVER_ROW = 256, 16  # the covers: one 16 px row per design, front edge (the spine) at the left


def book_atlas(seed: int):
    """Spine designs (cloth colour with grain, gilt bands, a title with lettering, an author mark,
    darker rounded edges) and a page-edge patch; and the matching covers (the same cloth with the
    hinge groove a few mm behind the spine). Deterministic."""
    from PIL import Image
    rng = np.random.default_rng(seed)
    img = np.zeros((ATLAS_H, ATLAS_W, 3), float)
    covers = np.zeros((COVER_ROW * (ATLAS_W // COL), COVER_W, 3), float)
    for c in range(ATLAS_W // COL - 1):
        base = np.array(BOOK_COLOURS[rng.integers(len(BOOK_COLOURS))], float) * rng.uniform(0.85, 1.1)
        col = np.tile(base, (ATLAS_H, COL, 1))
        col *= 1.0 + 0.06 * rng.standard_normal((ATLAS_H, COL, 1))  # cloth grain
        col[:, :1] *= 0.75  # rounded spine edges read darker
        col[:, -1:] *= 0.75
        gilt = np.array([196, 160, 82], float) if base.mean() < 150 else np.array([40, 30, 25], float)
        style = int(rng.integers(3))
        for frac in ((0.05, 0.95) if style != 2 else (0.08, 0.12, 0.88, 0.92)):
            r = int(frac * ATLAS_H)
            col[r:r + 2, 1:-1] = gilt
        t0, t1 = int(rng.uniform(0.18, 0.28) * ATLAS_H), int(rng.uniform(0.42, 0.55) * ATLAS_H)
        if style == 1:  # a contrasting title label
            col[t0:t1, 2:-2] = np.array(BOOK_COLOURS[rng.integers(len(BOOK_COLOURS))], float) * 0.9
        for r in range(t0 + 4, t1 - 4, 6):  # lettering: short gilt strokes
            w = int(rng.integers(3, COL - 5))
            col[r:r + 3, 3:3 + w] = gilt
        a = int(rng.uniform(0.72, 0.8) * ATLAS_H)
        col[a:a + 3, 4:COL - 4] = gilt  # author or publisher mark
        img[:, c * COL:(c + 1) * COL] = col
        cov = np.tile(base, (COVER_ROW, COVER_W, 1)) * (1.0 + 0.05 * rng.standard_normal((COVER_ROW, COVER_W, 1)))
        cov[:, :3] *= 0.8  # the rounded spine edge
        cov[:, 8:11] *= 0.62  # the hinge groove, about 8 mm behind the spine
        cov[:, 11:13] *= 1.08  # the board edge catching the light
        covers[c * COVER_ROW:(c + 1) * COVER_ROW] = cov
    paper = np.tile(np.array([228, 218, 196], float), (ATLAS_H, COL, 1))
    paper[:, ::2] *= 0.94  # page lines
    img[:, -COL:] = paper
    covers[-COVER_ROW:] = (228, 218, 196)
    return (Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)),
            Image.fromarray(np.clip(covers, 0, 255).astype(np.uint8)))


def _rot_y(deg):
    a = math.radians(deg)
    return np.array([[math.cos(a), 0.0, math.sin(a)], [0.0, 1.0, 0.0], [-math.sin(a), 0.0, math.cos(a)]])


def shelf_books(seed: int, tops=(0.64, 1.147, 1.653), clear=0.45) -> dict:
    """Rows of books for steel_frame_shelves_01 (its own frame: boards between x -0.517 and 0.517,
    front at -y), standing on the boards at heights `tops` with `clear` m free above each: hardbacks
    of varied height and depth, spines to the front with titles, front edges 0 to 3 cm back from
    the board edge, a few gaps, 2 to 4 books a row leaning 5 to 12 degrees on a neighbour, and on
    one row one or two books lying flat on top."""
    import random
    rnd = random.Random(seed)
    m = Mesh()
    designs = ATLAS_W // COL - 1
    pages = ((designs * COL + 1) / ATLAS_W, 0.0, (ATLAS_W - 1) / ATLAS_W, 1.0)

    def add_book(x, y, z, t, h, d, c, rot=None, pivot=(0.0, 0.0, 0.0)):
        u0, u1 = (c * COL + 1) / ATLAS_W, ((c + 1) * COL - 1) / ATLAS_W
        spine = (u0, 0.0, u1, 1.0)
        rows = ((c * COVER_ROW + 3) / (COVER_ROW * (designs + 1)), ((c + 1) * COVER_ROW - 3) / (COVER_ROW * (designs + 1)))
        front_left, front_right = (0.0, rows[0], 1.0, rows[1]), (1.0, rows[0], 0.0, rows[1])
        m.textured_box("book_cloth", (x, y, z), (x + t, y + d, z + h),
                       {"-y": spine, "+y": pages, "+z": pages, "+x": front_left, "-x": front_right,
                        "-z": front_left}, {"+x": "book_cover", "-x": "book_cover", "-z": "book_cover"},
                       rot, pivot)

    flat_row = rnd.randrange(len(tops))
    for row, top in enumerate(tops):
        leans = rnd.randint(2, 4)
        p_lean = leans / 30.0
        books = []  # standing books: (x, t, h, d, front y, design)
        x = -0.505
        last_lean = False
        while True:
            t = rnd.uniform(0.016, 0.042)
            if x + t > 0.505:
                break
            if rnd.random() < 0.06:  # a gap
                x += rnd.uniform(0.04, 0.12)
                last_lean = False
                continue
            h, d = rnd.uniform(0.17, 0.29), rnd.uniform(0.15, 0.23)
            y = -0.24 + rnd.uniform(0.0, 0.03)
            c = rnd.randrange(designs)
            if leans and not last_lean and rnd.random() < p_lean:
                phi = rnd.uniform(5.0, 12.0)
                g = h * math.sin(math.radians(phi)) + 0.001
                prev = books[-1] if books and abs(books[-1][0] + books[-1][1] - x) < 0.005 else None
                if prev is not None and prev[2] >= h * math.cos(math.radians(phi)) + 0.01 and rnd.random() < 0.5:
                    x += g  # leans left on the book before it, a gap opened under its top
                    add_book(x, y, top, t, h, d, c, _rot_y(-phi), (x, 0.0, top))
                    x += t + rnd.uniform(0.0, 0.003)
                    leans -= 1
                    last_lean = True
                    continue
                if x + t + g + 0.016 <= 0.505:  # leans right on the next book, made tall enough
                    add_book(x, y, top, t, h, d, c, _rot_y(phi), (x + t, 0.0, top))
                    x += t + g
                    t2 = rnd.uniform(0.016, min(0.042, 0.505 - x))
                    h2 = max(rnd.uniform(0.17, 0.29), min(0.29, h * math.cos(math.radians(phi)) + 0.02))
                    books.append((x, t2, h2, rnd.uniform(0.15, 0.23), -0.24 + rnd.uniform(0.0, 0.03),
                                  rnd.randrange(designs)))
                    x += t2 + rnd.uniform(0.0, 0.004)
                    leans -= 1
                    last_lean = True
                    continue
            books.append((x, t, h, d, y, c))
            x += t + rnd.uniform(0.0, 0.004)
            last_lean = False
        for bx, bt, bh, bd, by, bc in books:
            add_book(bx, by, top, bt, bh, bd, bc)
        if row != flat_row:
            continue
        for _ in range(rnd.randint(1, 2)):  # lying flat across the tops of a run of standing books
            L = rnd.uniform(0.2, 0.25)
            best = None
            for i in range(len(books)):
                run = [b for b in books if b[0] + b[1] > books[i][0] and b[0] < books[i][0] + L]
                if run[-1][0] + run[-1][1] < books[i][0] + L - 0.02:  # it would hang over the end
                    continue
                if any(n[0] - (p[0] + p[1]) > 0.006 for p, n in zip(run, run[1:])):  # or over a gap
                    continue
                spread = max(b[2] for b in run) - min(b[2] for b in run)
                if best is None or spread < best[0]:
                    best = (spread, books[i][0], max(b[2] for b in run), max(b[4] for b in run))
            if best is None:
                break
            _, bx, bh, by = best
            t, d, c = rnd.uniform(0.02, 0.035), rnd.uniform(0.15, 0.2), rnd.randrange(designs)
            z = top + bh + 0.0005
            if bh + t > clear:
                break
            # a standing book turned 90 degrees about y (its top to +x), resting on the run
            add_book(bx, by, z, t, L, d, c, _rot_y(90.0), (bx + t / 2, 0.0, z + t / 2))
            books = [b if not (b[0] + b[1] > bx and b[0] < bx + L) else (b[0], b[1], bh + t, b[3], b[4], b[5])
                     for b in books]
    spines, covers = book_atlas(seed)
    return m.write(f"proc_shelf_books_{seed}", "Books for a steel shelf (procedural)",
                   "visual only, on the shelf boards above the gate height",
                   {"book_cloth": spines, "book_cover": covers}, base_at_zero=False)


def _oak_leg(m, rnd, cx, cy, half, height, region="quarter_a"):
    """A square oak leg from the floor, the grain running up it: built lying along x (the atlas's
    grain direction) and stood up about y."""
    L, w = height, 2 * half
    m.textured_box("oak", (cx, cy - half, -half), (cx + L, cy + half, half),
                   {"-y": _oak(L, w, region, rnd), "+y": _oak(L, w, region, rnd),
                    "-z": _oak(L, w, region, rnd), "+z": _oak(L, w, region, rnd),
                    "-x": _oak(w, w, "end", rnd), "+x": _oak(w, w, "end", rnd)},
                   rot=_rot_y(-90.0), pivot=(cx, cy, 0.0))


def _box_seating(m: Mesh, hw: float, seats: int) -> None:
    """Box-arm seating 2 hw wide, 0.84 deep, 0.80 tall, upholstered in a woven oatmeal fabric: an
    upholstered base (0.06 to 0.30 m: it crosses the lidar plane) on a recessed dark plinth, so
    there is no space under it; rounded arms (0.16 m) and back with piped rims; between the arms
    `seats` loose piped seat cushions (seat 0.43 m) and back cushions reclined 10 degrees. Front -y."""
    m.box("plinth", (0, 0, 0.03), (hw - 0.02, 0.40, 0.03))
    m.rounded_box("upholstery", (0, 0, 0.18), (hw, 0.42, 0.12), 0.012)  # base
    # the back and the arms stand on the plinth beside the base, the cushions sit 2 cm into it, so
    # nothing below 0.40 m is rounded by more than 1.2 cm (its members are plain boxes); they stand
    # 3 mm proud of the base, so no face of theirs lies in one of its faces
    m.rounded_box("upholstery", (0, 0.3415, 0.43), (hw + 0.003, 0.0815, 0.37), 0.035, piping=("+z",),
                  pipe_mat="piping_oat")
    for sx in (-1, 1):  # arms
        m.rounded_box("upholstery", (sx * (hw - 0.0785), -0.0815, 0.34), (0.0815, 0.3415, 0.28), 0.035, piping=("+z",),
                      pipe_mat="piping_oat")
    inner = hw - 0.16
    for k in range(seats):
        x = -inner + (2 * k + 1) * inner / seats
        w = inner / seats - 0.005
        m.rounded_box("upholstery", (x, -0.075, 0.355), (w, 0.335, 0.075), 0.03, piping=("+z",),
                      pipe_mat="piping_oat")  # seat cushion
        m.rounded_box("upholstery", (x, 0.19, 0.59), (w, 0.06, 0.16), 0.03, pitch_deg=-10.0, piping=("-y",),
                      pipe_mat="piping_oat")  # back cushion


def armchair() -> dict:
    """Box-arm lounge chair, 0.82 x 0.84 x 0.80 m (_box_seating, one seat)."""
    m = Mesh()
    _box_seating(m, 0.41, 1)
    return m.write("proc_armchair", "Lounge armchair (procedural)", "members: plinth, base, back, arms, seat cushions")


def sofa() -> dict:
    """Three-seat box-arm sofa, 2.10 x 0.84 x 0.80 m (_box_seating, three seats), the armchair's pair."""
    m = Mesh()
    _box_seating(m, 1.05, 3)
    return m.write("proc_sofa", "Three-seat sofa (procedural)", "members: plinth, base, back, arms, seat cushions")


def ottoman() -> dict:
    """Upholstered cube ottoman, 0.60 x 0.60 x 0.42 m, the sofa's fabric: a base on a recessed dark
    plinth (nothing under it) and a piped top cushion."""
    m = Mesh()
    m.box("plinth", (0, 0, 0.03), (0.28, 0.28, 0.03))
    m.rounded_box("upholstery", (0, 0, 0.18), (0.30, 0.30, 0.12), 0.012)
    m.rounded_box("upholstery", (0, 0, 0.35), (0.30, 0.30, 0.07), 0.03, piping=("+z",), pipe_mat="piping_oat")
    return m.write("proc_ottoman", "Ottoman (procedural)", "members: plinth, base, back, arms, seat cushions")


COFFEE = {"w": 0.55, "d": 0.275, "h": 0.42, "top": 0.03, "leg": 0.025, "inset": 0.025}


def coffee_table() -> dict:
    """Oak coffee table, 1.10 x 0.55 x 0.42 m: a 3 cm flat-sawn top, a 6 cm apron set 3.5 cm in,
    four square 5 cm legs from the floor (set 2.5 cm in, 0.40 m apart front to back; nothing else
    below the apron). Long side along x."""
    import random
    rnd = random.Random(11)
    m = Mesh()
    c = COFFEE
    w, d, h, t = c["w"], c["d"], c["h"], c["top"]
    m.textured_box("oak", (-w, -d, h - t), (w, d, h),
                   {"+z": _oak(2 * w, 2 * d, "flat_a", rnd), "-z": _oak(2 * w, 2 * d, "flat_b", rnd),
                    "-y": _oak(2 * w, t, "edge", rnd), "+y": _oak(2 * w, t, "edge", rnd),
                    "-x": _oak(2 * d, t, "end", rnd), "+x": _oak(2 * d, t, "end", rnd)})
    ax, ay = w - 0.035, d - 0.035  # the apron's outer faces
    for sy in (-1, 1):  # long rails
        y0 = sy * ay - (0.02 if sy > 0 else 0.0)
        m.textured_box("oak", (-ax, y0, h - t - 0.06), (ax, y0 + 0.02, h - t),
                       {"-y": _oak(2 * ax, 0.06, "quarter_a", rnd), "+y": _oak(2 * ax, 0.06, "quarter_a", rnd),
                        "-z": _oak(2 * ax, 0.02, "quarter_a", rnd), "+z": _oak(2 * ax, 0.02, "quarter_a", rnd),
                        "-x": _oak(0.02, 0.06, "end", rnd), "+x": _oak(0.02, 0.06, "end", rnd)})
    for sx in (-1, 1):  # short rails
        x0 = sx * ax - (0.02 if sx > 0 else 0.0)
        m.textured_box("oak", (x0, -ay, h - t - 0.06), (x0 + 0.02, ay, h - t),
                       {"-x": _oak(2 * ay, 0.06, "quarter_b", rnd), "+x": _oak(2 * ay, 0.06, "quarter_b", rnd),
                        "-z": _oak(2 * ay, 0.02, "quarter_b", rnd), "+z": _oak(2 * ay, 0.02, "quarter_b", rnd),
                        "-y": _oak(0.02, 0.06, "end", rnd), "+y": _oak(0.02, 0.06, "end", rnd)})
    lx, ly = w - c["inset"] - c["leg"], d - c["inset"] - c["leg"]
    for sx in (-1, 1):
        for sy in (-1, 1):
            _oak_leg(m, rnd, sx * lx, sy * ly, c["leg"], h - t)
    return m.write("proc_coffee_table", "Coffee table (procedural)", "members: top, apron, four legs",
                   {"oak": oak_texture()})


def end_table() -> dict:
    """Closed oak end table, 0.45 x 0.45 x 0.55 m: a quarter-sawn body on a recessed dark plinth
    (no space under it), a 2 cm flat-sawn top with a 1 cm overhang, a drawer at the front over a
    dark gap with a small brass knob. Front faces -y."""
    import random
    rnd = random.Random(12)
    m = Mesh()
    b, t = 0.22, 0.23
    m.box("steel_dark", (0, 0, 0.015), (0.205, 0.205, 0.015))  # plinth
    m.textured_box("oak", (-b, -b, 0.03), (b, b, 0.53),
                   {"-y": _oak(2 * b, 0.5, "flat_b", rnd), "+y": _oak(2 * b, 0.5, "flat_a", rnd),
                    "-x": _oak(2 * b, 0.5, "quarter_a", rnd), "+x": _oak(2 * b, 0.5, "quarter_b", rnd),
                    "+z": _oak(2 * b, 2 * b, "flat_b", rnd), "-z": _oak(2 * b, 2 * b, "flat_b", rnd)})
    m.textured_box("oak", (-t, -t, 0.53), (t, t, 0.55),
                   {"+z": _oak(2 * t, 2 * t, "flat_a", rnd), "-z": _oak(2 * t, 2 * t, "flat_a", rnd),
                    "-y": _oak(2 * t, 0.02, "edge", rnd), "+y": _oak(2 * t, 0.02, "edge", rnd),
                    "-x": _oak(2 * t, 0.02, "end", rnd), "+x": _oak(2 * t, 0.02, "end", rnd)})
    m.box("shadow", (0, -b - 0.0005, 0.45), (0.193, 0.0005, 0.053))  # the drawer's gap
    m.textured_box("oak", (-0.19, -b - 0.004, 0.4), (0.19, -b - 0.001, 0.5),
                   {"-y": _oak(0.38, 0.1, "flat_a", rnd), "+y": _oak(0.38, 0.1, "flat_a", rnd),
                    "-z": _oak(0.38, 0.003, "edge", rnd), "+z": _oak(0.38, 0.003, "edge", rnd),
                    "-x": _oak(0.1, 0.003, "end", rnd), "+x": _oak(0.1, 0.003, "end", rnd)})
    m.cylinder("brass", (0, -b - 0.004 - 0.008, 0.45), 0.011, 0.008, axis="y", seg=20)  # knob
    for sx in (-1, 1):  # the mitre joints
        for sy in (-1, 1):
            m.box("shadow", (sx * b, sy * b, 0.28), (0.0007, 0.0007, 0.25), yaw_deg=45.0)
    return m.write("proc_end_table", "End table (procedural)", "members: plinth, body, top",
                   {"oak": oak_texture()})


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for fn in (office_chair, filing_cabinet, waste_bin, floor_lamp, plant_stand, shelf_feet, lambda: shelf_books(11),
               armchair, sofa, ottoman, coffee_table, end_table):
        rec = fn()
        faces = sum(p["faces"] for p in rec["parts"])
        print(f"{rec['id']:22s} {faces:5d} faces  size {rec['size_m']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
