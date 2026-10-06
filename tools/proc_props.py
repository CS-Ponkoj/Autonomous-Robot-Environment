"""Procedural props (small everyday objects without a fitting CC0 model), written like the
procedural furniture: <id>.json plus one binary .msh (and texture) per material in
robot_env/assets/furniture/. Deterministic: seeded generators only.

    .venv\\Scripts\\python tools\\proc_props.py

Cartons: regular slotted cartons (corrugated board, flaps meeting along the top, packing tape over
the seam and 6 cm down each end, a printed handling mark, some with a shipping label), textured
per face at real scale. Pallet: a block pallet (bottom boards, 9 blocks, stringer boards, top deck
boards with gaps), sawn pine with nail heads.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import proc_furniture as pf  # noqa: E402

pf.MATERIALS.update({
    "carton": ((1.0, 1.0, 1.0, 1.0), 0.06, 0.1, 0.0),  # corrugated board: matte, a little sheen
    "pine": ((1.0, 1.0, 1.0, 1.0), 0.08, 0.12, 0.0),  # sawn pallet pine
})

PPM = 650  # texture pixels per metre on the carton faces (about 1.5 mm per pixel)
CARD = np.array([171.0, 135.0, 92.0])  # kraft board
TAPE = np.array([205.0, 162.0, 104.0])  # tan packing tape: a little lighter and warmer than the board
INK = np.array([62.0, 46.0, 34.0])  # brown printing ink


def _noise(rng, h, w, beta, stretch=1.0):
    """Smooth noise of shape (h, w), zero mean, unit spread (power ~ 1/f^beta)."""
    fx = np.fft.fftfreq(w)[None, :] / stretch
    fy = np.fft.fftfreq(h)[:, None]
    f = np.sqrt(fx * fx + fy * fy)
    f[0, 0] = 1.0
    n = np.real(np.fft.ifft2(np.fft.fft2(rng.standard_normal((h, w))) / f ** (beta / 2)))
    return (n - n.mean()) / (n.std() + 1e-12)


def _board(rng, w, h):
    """Corrugated board seen from outside: kraft colour, mottling, fibres, faint flute ribs
    (vertical, 4 mm), darker handled edges and a few scuffs."""
    yy, xx = np.mgrid[0:h, 0:w].astype(float)
    k = 1.0 + 0.045 * _noise(rng, h, w, 2.4) + 0.025 * _noise(rng, h, w, 0.6)
    k += 0.012 * np.sin(2 * np.pi * xx / (0.004 * PPM))  # flutes under the liner
    edge = np.minimum(np.minimum(xx, w - 1 - xx), np.minimum(yy, h - 1 - yy)) / PPM
    k *= 1.0 - 0.18 * np.exp(-edge / 0.008)  # handled edges
    img = CARD[None, None, :] * k[..., None]
    for _ in range(rng.integers(1, 4)):  # scuffs: short pale streaks
        cx, cy = rng.uniform(0, w), rng.uniform(0, h)
        L, ang = rng.uniform(0.02, 0.07) * PPM, rng.uniform(0, math.pi)
        d = np.abs((xx - cx) * math.sin(ang) - (yy - cy) * math.cos(ang))
        t = (xx - cx) * math.cos(ang) + (yy - cy) * math.sin(ang)
        m = (d < 1.5) & (np.abs(t) < L / 2)
        img[m] = img[m] * 0.85 + 255 * 0.15
    return img


def _tape(img, x0, y0, x1, y1, rng):
    """Packing tape over a rectangle: tan, mostly opaque, with lengthwise sheen streaks."""
    h, w = img.shape[:2]
    x0, x1 = max(int(x0), 0), min(int(x1), w)
    y0, y1 = max(int(y0), 0), min(int(y1), h)
    if x1 <= x0 or y1 <= y0:
        return
    region = img[y0:y1, x0:x1]
    along_x = (x1 - x0) >= (y1 - y0)
    n = _noise(rng, y1 - y0, x1 - x0, 1.6, stretch=8.0 if along_x else 0.125)
    sheen = 1.0 + 0.05 * n
    region[:] = (0.45 * region + 0.55 * TAPE[None, None, :]) * sheen[..., None]  # the board shows through
    edge = np.zeros(region.shape[:2], bool)  # slightly darker tape edges
    if along_x:
        edge[:2, :], edge[-2:, :] = True, True
    else:
        edge[:, :2], edge[:, -2:] = True, True
    region[edge] *= 0.9
    if along_x:  # the tape's thickness catches the light along one edge
        region[2, :] = np.minimum(region[2, :] * 1.12, 255)
    else:
        region[:, 2] = np.minimum(region[:, 2] * 1.12, 255)


def _arrows(draw, x, y, s, color):
    """The 'this way up' handling mark: two upright arrows over a bar, in a thin frame."""
    c = tuple(int(v) for v in color)
    w = int(1.6 * s)
    draw.rectangle([x, y, x + w, y + int(1.3 * s)], outline=c, width=max(2, s // 14))
    for k in (0.45, 1.15):
        cx = x + int(k * s)
        draw.polygon([(cx, y + int(0.15 * s)), (cx - int(0.22 * s), y + int(0.45 * s)),
                      (cx + int(0.22 * s), y + int(0.45 * s))], fill=c)
        draw.rectangle([cx - int(0.06 * s), y + int(0.45 * s), cx + int(0.06 * s), y + int(1.0 * s)], fill=c)
    draw.rectangle([x + int(0.15 * s), y + int(1.08 * s), x + w - int(0.15 * s), y + int(1.16 * s)], fill=c)


def _glass(draw, x, y, s, color):
    """The 'fragile' mark: a stemmed glass in a thin frame."""
    c = tuple(int(v) for v in color)
    draw.rectangle([x, y, x + s, y + int(1.3 * s)], outline=c, width=max(2, s // 14))
    cx = x + s // 2
    draw.polygon([(cx - int(0.28 * s), y + int(0.2 * s)), (cx + int(0.28 * s), y + int(0.2 * s)),
                  (cx + int(0.12 * s), y + int(0.62 * s)), (cx - int(0.12 * s), y + int(0.62 * s))], fill=c)
    draw.rectangle([cx - int(0.03 * s), y + int(0.62 * s), cx + int(0.03 * s), y + int(1.0 * s)], fill=c)
    draw.rectangle([cx - int(0.2 * s), y + int(1.0 * s), cx + int(0.2 * s), y + int(1.07 * s)], fill=c)


def _label(img, x, y, rng):
    """A white shipping label (10 x 7 cm): a barcode and grey lines of print, no words."""
    w, h = int(0.10 * PPM), int(0.07 * PPM)
    lab = np.full((h, w, 3), 236.0) * (1.0 + 0.015 * _noise(rng, h, w, 1.0))[..., None]
    pil = Image.fromarray(np.clip(lab, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(pil)
    bx = int(0.08 * w)
    while bx < int(0.92 * w):  # barcode
        bw = int(rng.integers(1, 4))
        d.rectangle([bx, int(0.62 * h), bx + bw - 1, int(0.9 * h)], fill=(25, 25, 25))
        bx += bw + int(rng.integers(1, 4))
    for row in range(4):  # print lines of varied length
        ly = int((0.1 + 0.11 * row) * h)
        d.rectangle([int(0.08 * w), ly, int((0.3 + 0.6 * rng.random()) * w), ly + max(2, h // 30)], fill=(70, 70, 70))
    region = img[y:y + h, x:x + w]
    region[:] = np.asarray(pil, float)[:region.shape[0], :region.shape[1]]


def carton_texture(L, W, H, seed, label=True, fragile=False):
    """Texture atlas for a carton L x W x H (m): rows top (L x W), side (L x H), end (W x H).
    Returns the image and the uv rectangles (u0, v0, u1, v1) per face."""
    rng = np.random.default_rng(seed)
    pw, pl, ph = int(W * PPM), int(L * PPM), int(H * PPM)
    width = max(pl, pw)
    rows = [("top", pl, pw), ("side", pl, ph), ("side_b", pl, ph), ("end", pw, ph)]  # side_b: the far long side
    total = sum(r[2] for r in rows)
    atlas = np.zeros((total, width, 3))
    rects, y = {}, 0
    tape_w = 0.048 * PPM
    tail = 0.06 * PPM
    for name, w, h in rows:
        img = _board(rng, w, h)
        if name == "top":  # flaps meet along the length, tape over the seam
            img[h // 2 - 1:h // 2 + 1, :] *= 0.55
            _tape(img, 0, h / 2 - tape_w / 2, w, h / 2 + tape_w / 2, rng)
        elif name in ("side", "side_b"):
            img[:max(2, int(0.003 * PPM)), :] *= 0.8  # the fold of the top flap
            pil = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
            d = ImageDraw.Draw(pil)
            s = int(0.045 * PPM)
            ax = int(0.06 * PPM) if name == "side" else int(w - 0.06 * PPM - 1.6 * s)  # far side: top right
            _arrows(d, ax, int(0.05 * PPM), s, INK)
            if fragile and name == "side":
                _glass(d, ax + int(1.9 * s), int(0.05 * PPM), s, INK)
            img = np.asarray(pil.filter(ImageFilter.GaussianBlur(0.6)), float)
            if label and name == "side" and w > 0.2 * PPM and h > 0.12 * PPM:
                _label(img, int(w - 0.15 * PPM), int(h - 0.12 * PPM), rng)
        else:  # end: the tape tail runs 6 cm down from the top seam
            img[:max(2, int(0.003 * PPM)), :] *= 0.8
            _tape(img, w / 2 - tape_w / 2, 0, w / 2 + tape_w / 2, tail, rng)
        atlas[y:y + h, :w] = img
        rects[name] = (0.0, y / total, w / width, (y + h) / total)
        y += h
    return Image.fromarray(np.clip(atlas, 0, 255).astype(np.uint8)), rects


def carton(m: pf.Mesh, center, size, yaw_deg, rects, material="carton"):
    """A carton (L along x, W along y before the yaw) standing at center (base centre)."""
    L, W, H = size
    u_top, u_side, u_side_b, u_end = rects["top"], rects["side"], rects["side_b"], rects["end"]

    def gl(r):  # image rows (top = 0) to texture v (OpenGL: v = 0 at the image's bottom row)
        return (r[0], 1.0 - r[1], r[2], 1.0 - r[3])
    face_uv = {"+z": gl(u_top), "-z": gl(u_top), "-y": gl(u_side), "+y": gl(u_side_b), "-x": gl(u_end), "+x": gl(u_end)}
    c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    cx, cy, cz = center
    m.textured_box(material, (cx - L / 2, cy - W / 2, cz), (cx + L / 2, cy + W / 2, cz + H), face_uv,
                   rot=rot, pivot=(cx, cy, cz))


def pine_texture(seed, size=512):
    """Sawn pallet pine: grain along u (the board's length), knots, a little grey weathering."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size].astype(float)
    base = np.array([158.0, 134.0, 100.0])  # used pallet pine: grey-brown
    grain = np.sin(2 * np.pi * (yy / size * 22 + 0.6 * _noise(rng, size, size, 2.6, stretch=10.0) * 0.25))
    k = 1.0 + 0.04 * grain + 0.06 * _noise(rng, size, size, 2.0, stretch=6.0) + 0.03 * _noise(rng, size, size, 0.4)
    img = base[None, None, :] * k[..., None]
    grey = np.clip(0.5 + 0.5 * _noise(rng, size, size, 3.0), 0, 1)[..., None] * 0.32
    img = img * (1 - grey) + np.array([130.0, 128.0, 122.0]) * grey
    stain = np.clip(_noise(rng, size, size, 2.8) - 0.8, 0, None)[..., None]  # dirt and water stains
    img = img * (1.0 - 0.35 * np.minimum(stain, 1.0))
    for _ in range(3):  # knots
        cx, cy, r = rng.uniform(0, size), rng.uniform(0, size), rng.uniform(5, 11)
        d = np.hypot((xx - cx) / 2.2, yy - cy)
        img *= (1.0 - 0.35 * np.exp(-(d / r) ** 2))[..., None]
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))


def _plank(m: pf.Mesh, lo, hi, axis, seed_off, nails=()):
    """A sawn board from lo to hi with the grain along `axis` (0: x, 1: y); u runs along the grain
    (1 texture tile per 0.5 m), offset per board so neighbours differ. nails: (along, across) points
    on the top face given in metres from the board's centre, drawn as dark heads."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    tile = 0.5
    off = (seed_off * 0.37) % 1.0
    voff = (seed_off * 0.23) % 1.0  # rows of the texture differ per board too (no repeated stripes)
    other = 1 - axis

    def r(a0, a1, b0, b1):  # (u0, v0, u1, v1) for a face spanning along-axis a and across b
        return (a0 / tile + off, b0 / tile + voff, a1 / tile + off, b1 / tile + voff)
    face_uv = {
        "+z": r(lo[axis], hi[axis], lo[other], hi[other]), "-z": r(lo[axis], hi[axis], lo[other], hi[other]),
    }
    side = r(lo[axis], hi[axis], lo[2], hi[2])
    end = (lo[other] / tile + off, lo[2] / tile, hi[other] / tile + off, hi[2] / tile)
    if axis == 0:
        face_uv.update({"-y": side, "+y": side, "-x": end, "+x": end})
    else:
        face_uv.update({"-x": side, "+x": side, "-y": end, "+y": end})
    m.textured_box("pine", tuple(lo), tuple(hi), face_uv)
    for a, b in nails:  # nail heads, 1 mm proud
        p = (lo + hi) / 2
        p[axis] += a
        p[other] += b
        m.cylinder("nail", (p[0], p[1], hi[2] + 0.0005), 0.004, 0.0005, seg=8)


def pallet(m: pf.Mesh, L=0.8, W=0.7):
    """A block pallet L (x) by W (y), 144 mm tall: 3 bottom boards (22 mm) along x, 9 blocks (78 mm),
    3 stringer boards (22 mm) along x, 7 top deck boards (22 mm) along y with gaps. Base at z = 0,
    centre at the origin."""
    t, blk = 0.022, 0.078
    bw = 0.1  # board and block width
    ys = (-W / 2 + bw / 2, 0.0, W / 2 - bw / 2)
    xs = (-L / 2 + bw / 2, 0.0, L / 2 - bw / 2)
    k = 0
    for y in ys:  # bottom boards
        _plank(m, (-L / 2, y - bw / 2, 0.0), (L / 2, y + bw / 2, t), 0, k)
        k += 1
    for x in xs:  # blocks (grain upright: treat as boards along x, short)
        for y in ys:
            bx = 0.145 if x == 0.0 else bw
            _plank(m, (x - bx / 2, y - bw / 2, t), (x + bx / 2, y + bw / 2, t + blk), 0, k)
            k += 1
    for y in ys:  # stringer boards
        _plank(m, (-L / 2, y - bw / 2, t + blk), (L / 2, y + bw / 2, 2 * t + blk), 0, k)
        k += 1
    n = 7
    gap = (L - n * bw) / (n - 1)  # about 1.7 cm
    for i in range(n):  # top deck boards along y, nailed over the three stringers
        x0 = -L / 2 + i * (bw + gap)
        nails = [(y, d) for y in ys for d in (-0.025, 0.025)]
        _plank(m, (x0, -W / 2, 2 * t + blk), (x0 + bw, W / 2, 3 * t + blk), 1, k, nails)
        k += 1
    return 3 * t + blk


# The storage layouts (each model's frame: base centre at the origin, x along its length).
PALLET_STACK = {  # a half-height pallet load: 2 x 2 cartons in 2 layers
    "pallet": (0.8, 0.7),
    "carton": (0.392, 0.342, 0.29),
}


def atlas(tiles, cols, scale=1.0):
    """Pack carton textures (image, uv rects) into one image, `cols` per row; returns the image and
    each tile's rects remapped into it (one material and one texture for a whole load). `scale`
    shrinks the packed image (the rects are fractions, so they hold) to keep texture memory in budget."""
    tw = max(img.width for img, _ in tiles)
    th = max(img.height for img, _ in tiles)
    rows = math.ceil(len(tiles) / cols)
    out = Image.new("RGB", (tw * cols, th * rows), tuple(int(c) for c in CARD))
    mapped = []
    for k, (img, rects) in enumerate(tiles):
        ox, oy = (k % cols) * tw, (k // cols) * th
        out.paste(img, (ox, oy))
        sx, sy = img.width / out.width, img.height / out.height
        mapped.append({n: (ox / out.width + r[0] * sx, oy / out.height + r[1] * sy,
                           ox / out.width + r[2] * sx, oy / out.height + r[3] * sy) for n, r in rects.items()})
    if scale != 1.0:
        out = out.resize((round(out.width * scale), round(out.height * scale)), Image.LANCZOS)
    return out, mapped


def storage_pallet() -> dict:
    """A block pallet loaded with 8 cartons (2 x 2, 2 layers), each a few mm apart and turned up
    to 1.5 degrees, so the load reads as hand-stacked."""
    m = pf.Mesh()
    top = pallet(m, *PALLET_STACK["pallet"])
    L, W, H = PALLET_STACK["carton"]
    rnd = np.random.default_rng(31)
    tiles = [carton_texture(L, W, H, 100 + k, label=(k % 3 == 0), fragile=(k == 5)) for k in range(8)]
    img, rects = atlas(tiles, 4, scale=0.7)
    for layer in range(2):
        for i, sx in enumerate((-1, 1)):
            for j, sy in enumerate((-1, 1)):
                k = layer * 4 + i * 2 + j
                cx = sx * (L / 2 + 0.003) + rnd.uniform(-0.003, 0.003)
                cy = sy * (W / 2 + 0.003) + rnd.uniform(-0.003, 0.003)
                carton(m, (cx, cy, top + layer * H), (L, W, H), rnd.uniform(-1.5, 1.5), rects[k])
    return m.write("proc_storage_pallet", "Pallet with cartons (procedural)",
                   "members: bottom boards, blocks, stringer boards, deck and load",
                   {"pine": pine_texture(7), "carton": img})


def storage_carton() -> dict:
    """A large carton (60 x 58 x 32 cm) on the floor: the base of the corner stack."""
    m = pf.Mesh()
    L, W, H = 0.6, 0.58, 0.32
    img, rects = carton_texture(L, W, H, 201, label=True)
    carton(m, (0.0, 0.0, 0.0), (L, W, H), 0.0, rects)
    return m.write("proc_storage_carton", "Large carton (procedural)", "member: one box", {"carton": img})


pf.MATERIALS.setdefault("nail", ((0.24, 0.24, 0.25, 1.0), 0.4, 0.4, 0.0))
pf.MATERIALS.update({
    "clock_case": ((0.06, 0.06, 0.065, 1.0), 0.45, 0.55, 0.0),  # black moulded plastic
    "clock_dial": ((1.0, 1.0, 1.0, 1.0), 0.15, 0.2, 0.0),
    "clock_hand": ((0.05, 0.05, 0.055, 1.0), 0.3, 0.4, 0.0),
    "clock_second": ((0.75, 0.08, 0.06, 1.0), 0.3, 0.4, 0.0),
})


def dial_texture(size=512):
    """An office clock dial: warm white, 60 minute ticks, 12 hour bars (thicker at 12, 3, 6, 9),
    a printed inner ring; drawn at 4x and reduced (smooth edges), no lettering."""
    big = size * 4
    img = Image.new("RGB", (big, big), (238, 236, 230))
    d = ImageDraw.Draw(img)
    c = big / 2
    for k in range(60):
        a = 2 * math.pi * k / 60
        hour = k % 5 == 0
        quarter = k % 15 == 0
        r0 = c * (0.70 if hour else 0.86)
        r1 = c * 0.94
        w = c * (0.045 if quarter else 0.03 if hour else 0.008)
        sx, sy = math.sin(a), -math.cos(a)
        px, py = -sy * w / 2, sx * w / 2
        pts = [(c + sx * r0 + px, c + sy * r0 + py), (c + sx * r1 + px, c + sy * r1 + py),
               (c + sx * r1 - px, c + sy * r1 - py), (c + sx * r0 - px, c + sy * r0 - py)]
        d.polygon(pts, fill=(28, 28, 30))
    d.ellipse([c - c * 0.975, c - c * 0.975, c + c * 0.975, c + c * 0.975], outline=(60, 60, 62), width=int(c * 0.012))
    img = img.resize((size, size), Image.LANCZOS)
    a = np.asarray(img, float)  # a faint shading toward the rim (the case's shadow on the dial)
    yy, xx = np.mgrid[0:size, 0:size]
    r = np.hypot(xx - size / 2, yy - size / 2) / (size / 2)
    a *= (1.0 - 0.12 * np.clip((r - 0.85) / 0.15, 0, 1))[..., None]
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def wall_clock() -> dict:
    """A 30 cm office wall clock, 4 cm deep: a black case with a rounded rim, the dial 8 mm behind
    the rim's front, hour, minute and red second hands (at 10:09:30). Its back is at y = 0 and it
    faces -y; base (lowest point) at z = 0."""
    m = pf.Mesh()
    R, depth = 0.15, 0.04
    seg = 64
    ang = 2 * math.pi * np.arange(seg) / seg
    ring = np.stack([np.cos(ang), np.sin(ang)], 1)

    def to_world(p):  # local (x, up, out of the dial) -> world (x, -out, up); the dial faces -y
        p = np.asarray(p, float)
        return np.stack([p[..., 0], -p[..., 2], p[..., 1]], -1)
    # case side: a cylinder wall from the back (out = 0) to the front (out = depth), then the rim:
    # a ring 1.2 cm wide stepping in, and a short inner wall down to the dial
    profile = [(R - 0.004, 0.0), (R, 0.004), (R, depth - 0.006), (R - 0.004, depth), (R - 0.012, depth),
               (R - 0.014, depth - 0.008)]  # (radius, out) along the case from back to dial
    for (r0, z0), (r1, z1) in zip(profile, profile[1:]):
        v = np.vstack([np.column_stack([ring * r0, np.full(seg, z0)]), np.column_stack([ring * r1, np.full(seg, z1)])])
        dr, dz = r1 - r0, z1 - z0
        nrm = np.column_stack([ring * dz, np.full(seg, -dr)])  # outward normal of this band
        nrm /= np.maximum(np.linalg.norm(nrm, axis=1), 1e-12)[:, None]
        f = []
        for i in range(seg):
            j = (i + 1) % seg
            f += [(i, j, seg + j), (i, seg + j, seg + i)]
        m.add("clock_case", to_world(v), f, to_world(np.vstack([nrm, nrm])))
    back = np.vstack([[0.0, 0.0, 0.0], np.column_stack([ring * (R - 0.004), np.zeros(seg)])])
    m.add("clock_case", to_world(back), [(0, 1 + (i + 1) % seg, 1 + i) for i in range(seg)],
          to_world(np.tile([0.0, 0.0, -1.0], (seg + 1, 1))))
    rd = R - 0.014  # the dial
    zd = depth - 0.008
    pts = np.vstack([[0.0, 0.0], ring * rd])
    v = np.column_stack([pts, np.full(len(pts), zd)])
    uv = np.column_stack([(pts[:, 0] / rd + 1) / 2, (pts[:, 1] / rd + 1) / 2])  # v up (OpenGL)
    m.add("clock_dial", to_world(v), [(0, 1 + i, 1 + (i + 1) % seg) for i in range(seg)],
          to_world(np.tile([0.0, 0.0, 1.0], (seg + 1, 1))), uv)

    def hand(mat, angle_deg, length, width, tail, z, thick=0.0015):
        a = math.radians(angle_deg)
        ux, uy = math.sin(a), math.cos(a)  # clockwise from 12
        px, py = uy, -ux
        corners = []
        for s_len, s_w in ((-tail, width), (length, width * 0.6)):
            corners += [(ux * s_len + px * s_w / 2, uy * s_len + py * s_w / 2),
                        (ux * s_len - px * s_w / 2, uy * s_len - py * s_w / 2)]
        q = [corners[0], corners[2], corners[3], corners[1]]  # a tapered plate
        top = [(x, y, z + thick) for x, y in q]
        bot = [(x, y, z) for x, y in q]
        v = np.array(top + bot)
        f = [(0, 1, 2), (0, 2, 3), (4, 6, 5), (4, 7, 6)]
        for i in range(4):
            j = (i + 1) % 4
            f += [(i, 4 + i, 4 + j), (i, 4 + j, j)]
        m.add(mat, to_world(v), f)
    hand("clock_hand", 300 + 9 * 0.5, 0.075, 0.009, 0.015, zd + 0.002)  # 10:09
    hand("clock_hand", 9.5 * 6, 0.112, 0.007, 0.018, zd + 0.004)  # 9 min 30 s
    hand("clock_second", 30 * 6, 0.118, 0.0022, 0.03, zd + 0.006, thick=0.001)
    m.cylinder("clock_hand", (0.0, 0.0, zd + 0.0065), 0.006, 0.0015, seg=16)  # the hub, built in the dial's frame
    v, f, n, _ = m.parts["clock_hand"].pop()
    m.add("clock_hand", to_world(v), f, to_world(n))  # turned to face -y like the rest
    return m.write("proc_wall_clock", "Office wall clock (procedural)", "visual only, on a wall above the robot",
                   {"clock_dial": dial_texture()})



pf.MATERIALS.update({
    "monitor_body": ((0.07, 0.07, 0.075, 1.0), 0.35, 0.45, 0.0),  # matte black plastic
    "monitor_stand": ((0.16, 0.16, 0.17, 1.0), 0.5, 0.55, 0.0),  # dark grey metal
    "key_case": ((0.11, 0.11, 0.12, 1.0), 0.3, 0.35, 0.0),
    "keycap": ((1.0, 1.0, 1.0, 1.0), 0.25, 0.3, 0.0),
    "mouse": ((0.09, 0.09, 0.1, 1.0), 0.45, 0.6, 0.0),
    "mouse_pad": ((0.12, 0.13, 0.15, 1.0), 0.05, 0.1, 0.0),
    "ceramic": ((0.9, 0.89, 0.86, 1.0), 0.55, 0.7, 0.0),
    "coffee": ((0.16, 0.09, 0.05, 1.0), 0.6, 0.8, 0.0),
    "paper": ((1.0, 1.0, 1.0, 1.0), 0.05, 0.1, 0.0),
    "pen": ((0.1, 0.18, 0.55, 1.0), 0.5, 0.6, 0.0),
    "pen_cap": ((0.05, 0.1, 0.35, 1.0), 0.5, 0.6, 0.0),
    "pen_tip": ((0.04, 0.04, 0.04, 1.0), 0.6, 0.6, 0.0),
    "monitor_logo": ((0.3, 0.3, 0.32, 1.0), 0.6, 0.6, 0.0),
    "screen": ((1.0, 1.0, 1.0, 1.0), 0.7, 0.9, 0.0),
    "cable": ((0.05, 0.05, 0.05, 1.0), 0.3, 0.3, 0.0),
})
MONITOR = {"width": 0.54, "height": 0.33, "bottom": 0.105, "front_y": 0.155}  # desk-set frame (build_world.DESK_MONITOR)


def screen_texture(w=540, h=330):
    """A switched-off monitor: near-black glass, a soft diagonal sheen from the upper left, and the
    room's window as a faint blurred light patch in the upper half (what an off panel mirrors)."""
    yy, xx = np.mgrid[0:h, 0:w].astype(float)
    base = 13.0 + 10.0 * np.clip(1.0 - (xx / w + yy / h) / 1.2, 0, 1) ** 2  # the sheen
    patch = np.zeros((h, w))
    patch[int(0.12 * h):int(0.55 * h), int(0.3 * w):int(0.75 * w)] = 1.0
    patch[int(0.12 * h):int(0.55 * h), int(0.515 * w):int(0.53 * w)] = 0.4  # the window's mullion
    img = Image.fromarray((patch * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(9))
    win = np.asarray(img, float) / 255.0
    rgb = np.stack([base + 14 * win, base + 16 * win, base + 20 * win], -1)
    return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))


def keycap_texture(cells=16, cell=32):
    """Keycap tops: dark grey with a slightly darker dish in the middle and, in each cell's upper left,
    an abstract light-grey mark (2 to 6 px bars, never letters); cell (0, 0) is left blank."""
    size = cells * cell
    img = Image.new("RGB", (size, size), (44, 44, 47))
    d = ImageDraw.Draw(img)
    rng = np.random.default_rng(78)
    for k in range(cells * cells):
        x0, y0 = (k % cells) * cell, (k // cells) * cell
        d.rectangle([x0 + 4, y0 + 4, x0 + cell - 5, y0 + cell - 5], fill=(40, 40, 43))  # the dish
        if k == 0:
            continue
        mx, my = x0 + 7, y0 + 7
        for _ in range(int(rng.integers(1, 4))):
            if rng.random() < 0.6:
                w, h = int(rng.integers(2, 7)), int(rng.integers(2, 4))
            else:
                w, h = int(rng.integers(2, 4)), int(rng.integers(3, 7))
            d.rectangle([mx, my, mx + w, my + h], fill=(168, 168, 172))
            mx += w + int(rng.integers(1, 3))
    return img


def notepad_texture(size=256):
    """Lined paper with a red margin and a few pencil strokes (no words)."""
    img = Image.new("RGB", (size, int(size * 1.4)), (229, 228, 221))
    d = ImageDraw.Draw(img)
    for k in range(1, 22):
        y = int(k * img.height / 22)
        d.line([(0, y), (size, y)], fill=(170, 190, 215), width=1)
    d.line([(int(0.14 * size), 0), (int(0.14 * size), img.height)], fill=(215, 120, 120), width=1)
    rng = np.random.default_rng(5)
    for k in range(3, 9):  # strokes on a few lines
        y = int(k * img.height / 22) - 3
        x = int(0.18 * size)
        while x < int((0.4 + 0.5 * rng.random()) * size):
            w = int(rng.integers(8, 26))
            d.line([(x, y + int(rng.integers(-1, 2))), (x + w, y + int(rng.integers(-1, 2)))], fill=(90, 90, 100), width=1)
            x += w + int(rng.integers(4, 9))
    return img


def desk_set() -> dict:
    """What stands on the office desk (frame: the desk-top centre, z = 0 on the top, front at -y):
    a 24-inch monitor (thin bezel, back housing, neck, oval base, cable to the desk) whose screen is
    a separate reflective box in the world (MONITOR), a keyboard with keycaps on a sloped case, a
    mouse on a pad, a mug of coffee, and a notepad with a pen, each turned a few degrees."""
    m = pf.Mesh()
    W, Hs, z0, fy = MONITOR["width"], MONITOR["height"], MONITOR["bottom"], MONITOR["front_y"]
    bez = 0.008
    zc = z0 + Hs / 2
    m.rounded_box("monitor_body", (0.0, fy + 0.012, zc), (W / 2 + bez, 0.012, Hs / 2 + bez), 0.004, n_arc=1)
    m.rounded_box("monitor_body", (0.0, fy + 0.04, zc - 0.02), (0.17, 0.018, 0.11), 0.015)  # back housing
    m.box("monitor_logo", (0.0, fy - 0.0002, z0 - bez / 2), (0.02, 0.0004, 0.0018))  # a small logo on the chin
    # the switched-off screen: a textured quad just in front of the bezel (an off panel's dark glass
    # with its sheen and the window's soft reflection baked in; MuJoCo mirrors no mesh)
    q = np.array([(-W / 2, fy - 0.0004, z0 + Hs), (W / 2, fy - 0.0004, z0 + Hs), (W / 2, fy - 0.0004, z0),
                  (-W / 2, fy - 0.0004, z0)])
    m.add("screen", q, [(0, 2, 1), (0, 3, 2)], np.tile([0.0, -1.0, 0.0], (4, 1)),
          np.array([(0.0, 1.0), (1.0, 1.0), (1.0, 0.0), (0.0, 0.0)]))
    m.box("monitor_stand", (0.0, fy + 0.07, 0.11), (0.03, 0.009, 0.1), pitch_deg=-8.0)  # neck, leaning back
    m.cushion("monitor_stand", (0.0, fy + 0.05, 0.006), (0.12, 0.09, 0.006), e=0.6)  # oval base
    path = np.array([(0.05, fy + 0.06, z) for z in np.linspace(0.12, 0.02, 6)] + [(0.08, fy + 0.12, 0.004)])
    out = np.tile([1.0, 0.0, 0.0], (len(path), 1))
    for a, b, oa in zip(path, path[1:], out):  # the cable: short straight tubes
        seg = np.array([a, b])
        mid = (a + b) / 2
        m.tube("cable", seg - mid, np.array([oa, oa]), 0.0035, mid)
    # keyboard: a sloped case (front 12 mm, back 22 mm) with 6 rows of keycaps on a 19 mm pitch
    kx, ky, kw, kd = -0.02, -0.13, 0.44, 0.135
    m.box("key_case", (kx, ky, 0.008), (kw / 2, kd / 2, 0.008))
    m.box("key_case", (kx, ky + 0.02, 0.016), (kw / 2, kd / 2 - 0.02, 0.006), pitch_deg=3.0)
    # rows from the front: the space-bar row, 4 letter/number rows, the function row; each a main
    # block (key widths in units of 19 mm), a gap, the arrow/navigation column, a gap, the numpad
    main = [[1.25, 1.25, 1.25, 6.25, 1.25, 1.25, 1.25, 1.25], [2.25] + [1.0] * 10 + [2.75],
            [1.75] + [1.0] * 11 + [2.25], [1.5] + [1.0] * 12 + [1.5], [1.0] * 13 + [2.0],
            [1.0] + [0.0] + [1.0] * 4 + [0.5] + [1.0] * 4 + [0.5] + [1.0] * 4]
    nav = [3, 1, 0, 0, 3, 3]  # keys in the 3-wide navigation column per row (arrows, home block)
    pad = [[2.0, 1.0, 1.0], [1.0] * 4, [1.0] * 4, [1.0] * 4, [1.0] * 4, []]
    unit = 0.019
    x0 = kx - (15 + 0.5 + 3 + 0.5 + 4) * unit / 2
    cells = 16  # the legend atlas (keycap_texture): 16 x 16 cells, cell (0, 0) blank for the sides
    keyrng = np.random.default_rng(77)
    blank = (0.0, 1.0, 1.0 / cells, 1.0 - 1.0 / cells)  # OpenGL v

    def cap(cx, cy, cz, hx):
        k = int(keyrng.integers(1, cells * cells))
        u0, v0 = (k % cells) / cells, (k // cells) / cells
        faces = {f: blank for f in ("-x", "+x", "-y", "+y", "-z")}
        faces["+z"] = (u0, 1.0 - v0, u0 + 1.0 / cells, 1.0 - v0 - 1.0 / cells) if hx < 0.012 else blank
        m.textured_box("keycap", (cx - hx, cy - 0.0077, cz - 0.0045), (cx + hx, cy + 0.0077, cz + 0.0045), faces)
    for r in range(6):
        yy = ky - kd / 2 + 0.014 + r * unit + (0.004 if r == 5 else 0.0)
        zz = 0.02 + 0.0011 * r
        x = x0
        for wk in main[r]:
            if wk > 0.0:
                cap(x + wk * unit / 2, yy, zz, wk * unit / 2 - 0.0018)
            x += (wk if wk > 0.0 else 1.0) * unit if wk != 0.5 else 0.5 * unit
        x = x0 + 15.5 * unit
        if nav[r] == 3:
            for k in range(3):
                cap(x + (k + 0.5) * unit, yy, zz, unit / 2 - 0.0018)
        elif nav[r] == 1:
            cap(x + 1.5 * unit, yy, zz, unit / 2 - 0.0018)
        x = x0 + 19 * unit
        for wk in pad[r]:
            cap(x + wk * unit / 2, yy, zz, wk * unit / 2 - 0.0018)
            x += wk * unit
    # mouse on its pad
    m.rounded_box("mouse_pad", (0.36, -0.12, 0.0015), (0.125, 0.105, 0.0015), 0.0014)
    m.cushion("mouse", (0.37, -0.115, 0.0185), (0.031, 0.056, 0.016), e=0.55, nu=24, nv=12)
    m.box("cable", (0.37, -0.075, 0.033), (0.0015, 0.006, 0.0012))  # the wheel
    # a mug of coffee: tapered ceramic, the coffee 1 cm below the rim, a handle
    mx, my = -0.42, 0.02
    m.cylinder("ceramic", (mx, my, 0.0475), 0.039, 0.0475, seg=28, caps=False, radius_top=0.042)
    m.cylinder("ceramic", (mx, my, 0.0475), 0.035, 0.0475, seg=28, caps=False, radius_top=0.038, inward=True)
    for mat, z, r0, r1, up in (("ceramic", 0.095, 0.038, 0.042, True), ("ceramic", 0.0, 0.0, 0.039, False),
                               ("coffee", 0.085, 0.0, 0.037, True)):
        m.annulus(mat, z, r0, r1, seg=28, up=up)  # annulus() is centred on the z axis: move it to the mug
        v, f, n, uv = m.parts[mat].pop()
        m.add(mat, v + [mx, my, 0.0], f, n, uv)
    t = np.radians(np.linspace(-80, 80, 9))
    hp = np.column_stack([np.full(9, mx + 0.04) + 0.028 * np.cos(t), np.full(9, my), 0.05 + 0.028 * np.sin(t)])
    ho = np.column_stack([np.cos(t), np.zeros(9), np.sin(t)])
    for a, b, oa in zip(hp, hp[1:], ho):
        mid = (a + b) / 2
        m.tube("ceramic", np.array([a, b]) - mid, np.array([oa, oa]), 0.006, mid)
    # notepad (A5) with a pen, turned 8 degrees
    c, s_ = math.cos(math.radians(8)), math.sin(math.radians(8))
    rot = np.array([[c, -s_, 0], [s_, c, 0], [0, 0, 1.0]])
    full = (0.0, 0.0, 1.0, 1.0)
    side = (0.0, 0.0, 0.02, 0.02)
    m.textured_box("paper", (0.26, 0.02, 0.0), (0.408, 0.23, 0.008),
                   {"+z": (0.0, 0.0, 1.0, 1.0), "-z": side, "-x": side, "+x": side, "-y": side, "+y": side},
                   rot=rot, pivot=(0.334, 0.125, 0.0))
    m.cylinder("pen", (0.0, 0.0, 0.0), 0.0045, 0.07, seg=10)
    v, f, n, _ = m.parts["pen"].pop()  # lay the pen down along x on the notepad, turned 20 degrees
    a = math.radians(20)
    R = np.array([[0, 0, 1.0], [0, 1.0, 0], [-1.0, 0, 0]])
    Z = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1.0]])
    m.add("pen", v @ R.T @ Z.T + [0.33, 0.12, 0.0125], f, n @ R.T @ Z.T)
    for mat, r, hh, along in (("pen_cap", 0.0055, 0.01, 0.06), ("pen_tip", 0.002, 0.004, -0.074)):
        m.cylinder(mat, (0.0, 0.0, 0.0), r, hh, seg=10, radius_top=(r if mat == "pen_cap" else 0.0006))
        v, f, n, _ = m.parts[mat].pop()
        flip = np.diag([1.0, 1.0, -1.0]) if mat == "pen_tip" else np.eye(3)  # the tip narrows outward
        m.add(mat, (v @ flip.T + [0, 0, along]) @ R.T @ Z.T + [0.33, 0.12, 0.0125], f, n @ flip.T @ R.T @ Z.T)
    return m.write("proc_desk_set", "Monitor, keyboard, mouse, mug, notepad (procedural)",
                   "visual only, on the office desk top", {"paper": notepad_texture(), "keycap": keycap_texture(), "screen": screen_texture()})


pf.MATERIALS.update({
    "board": ((1.0, 1.0, 1.0, 1.0), 0.45, 0.6, 0.0),  # melamine: a soft gloss, no emission
    "alu": ((0.62, 0.63, 0.64, 1.0), 0.7, 0.7, 0.02),  # anodised aluminium frame and tray
    "corner": ((0.2, 0.2, 0.21, 1.0), 0.3, 0.35, 0.0),  # moulded corner caps
    "marker_blue": ((0.1, 0.22, 0.6, 1.0), 0.4, 0.5, 0.0),
    "marker_red": ((0.7, 0.1, 0.08, 1.0), 0.4, 0.5, 0.0),
    "marker_black": ((0.06, 0.06, 0.07, 1.0), 0.4, 0.5, 0.0),
    "marker_blue_cap": ((0.05, 0.12, 0.38, 1.0), 0.4, 0.5, 0.0),
    "marker_red_cap": ((0.45, 0.05, 0.04, 1.0), 0.4, 0.5, 0.0),
    "marker_black_cap": ((0.02, 0.02, 0.02, 1.0), 0.4, 0.5, 0.0),
    "marker_cap": ((0.1, 0.1, 0.1, 1.0), 0.4, 0.5, 0.0),
    "felt": ((0.18, 0.18, 0.2, 1.0), 0.05, 0.1, 0.0),
    "eraser": ((0.25, 0.42, 0.7, 1.0), 0.3, 0.35, 0.0),
})
BOARD = (1.2, 0.9)  # m, the writing surface


def _stroke(d, pts, color, width, rng, jitter=1.0):
    """A marker line through pts with a little hand wobble and width variation."""
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        n = max(2, int(math.hypot(x1 - x0, y1 - y0) / 6))
        prev = (x0, y0)
        for k in range(1, n + 1):
            t = k / n
            p = (x0 + (x1 - x0) * t + rng.normal(0, jitter), y0 + (y1 - y0) * t + rng.normal(0, jitter))
            d.line([prev, p], fill=color, width=max(1, int(width + rng.normal(0, 0.6))))
            prev = p


def _scribble(d, x, y, length, color, rng, h=14.0):
    """A line of 'handwriting': connected loops of varied size, grouped into word-length runs with
    gaps, never letters."""
    cx = x
    slant = math.tan(math.radians(rng.uniform(-2.0, 2.0)))  # the line drifts a little
    while cx < x + length:
        run = rng.uniform(40, 110)
        size = h * rng.uniform(0.8, 1.2)  # per word
        pts = []
        t = 0.0
        while t < run:  # one loop at a time, each its own width and height
            period = rng.uniform(3.0, 8.0) * 2.0
            amp = size * rng.uniform(0.35, 1.0)
            r = rng.random()
            if r < 0.1:
                amp *= 1.6  # an ascender
            elif r < 0.2:
                amp = -0.8 * size  # a descender
            for k in range(1, 5):
                u = t + period * k / 4
                pts.append((cx + u, y + (cx + u - x) * slant - amp * math.sin(math.pi * k / 4) + rng.normal(0, 0.6)))
            t += period
        _stroke(d, pts, color, 2.2, rng, 0.3)
        cx += run + rng.uniform(14, 26)


def whiteboard_texture(px_per_m=900):
    """The board's face: off-white melamine with faint ghosts of erased writing, a box diagram with
    arrows (blue, black), a list of 'handwritten' lines with bullets, a red circled area."""
    W, H = int(BOARD[0] * px_per_m), int(BOARD[1] * px_per_m)
    rng = np.random.default_rng(42)
    base = np.full((H, W, 3), 246.0) * (1.0 + 0.006 * _noise(rng, H, W, 2.0))[..., None]
    img = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))
    ghost = ImageDraw.Draw(img)
    for _ in range(9):  # ghosts of erased writing: pale grey-blue smears
        _scribble(ghost, rng.uniform(40, W - 400), rng.uniform(60, H - 40), rng.uniform(150, 380), (226, 229, 236), rng)
    d = ImageDraw.Draw(img)
    blue, black, red = (35, 62, 150), (30, 30, 34), (180, 40, 35)
    boxes = [(70, 90, 270, 190), (400, 90, 600, 190), (235, 300, 435, 400)]
    for b in boxes:
        x0, y0, x1, y1 = b
        _stroke(d, [(x0, y0), (x1, y0 + 2), (x1 - 1, y1), (x0 + 2, y1 - 1), (x0, y0)], blue, 3.2, rng, 0.8)
        _scribble(d, x0 + 20, (y0 + y1) / 2 + 8, x1 - x0 - 50, black, rng, h=12)
    for (ax, ay), (bx, by) in (((270, 140), (400, 140)), ((170, 190), (285, 300)), ((500, 190), (385, 300))):
        _stroke(d, [(ax, ay), (bx, by)], black, 2.6, rng, 0.6)
        ang = math.atan2(by - ay, bx - ax)
        for s in (2.6, -2.6):
            _stroke(d, [(bx, by), (bx - 18 * math.cos(ang + s / 6 * 3.14), by - 18 * math.sin(ang + s / 6 * 3.14))],
                    black, 2.6, rng, 0.3)
    for k in range(6):  # a bulleted list on the right
        y = 120 + k * 62
        d.ellipse([700, y - 12, 710, y - 2], fill=blue)
        _scribble(d, 730, y, rng.uniform(180, 300), blue if k % 3 else black, rng, h=15)
    t = np.linspace(0, 2 * math.pi, 40)  # a red loop around the last two lines
    loop = [(850 + 190 * math.cos(a) + rng.normal(0, 2), 430 + 70 * math.sin(a) + rng.normal(0, 2)) for a in t]
    _stroke(d, loop, red, 3.0, rng, 0.8)
    _scribble(d, 80, 600, 420, black, rng, h=16)
    _scribble(d, 80, 670, 300, black, rng, h=16)
    return img.filter(ImageFilter.GaussianBlur(0.5))


def whiteboard() -> dict:
    """A 1.2 x 0.9 m whiteboard: the board in an aluminium frame with dark corner caps, 2 cm deep
    overall, a tray along the bottom with three markers lying in it and an eraser; back at y = 0,
    facing -y, base (the tray's underside) at z = 0."""
    m = pf.Mesh()
    W, H = BOARD
    fw, depth = 0.018, 0.018
    tray_h = 0.03
    z0 = tray_h + 0.01  # the board's lower edge above the tray's underside
    zc = z0 + H / 2
    m.textured_box("board", (-W / 2, -depth + 0.004, z0), (W / 2, 0.0, z0 + H),
                   {"-y": (0.0, 1.0, 1.0, 0.0), "+y": (0, 0, 0.01, 0.01), "-x": (0, 0, 0.01, 0.01),
                    "+x": (0, 0, 0.01, 0.01), "+z": (0, 0, 0.01, 0.01), "-z": (0, 0, 0.01, 0.01)})
    for x0, x1, za, zb in ((-W / 2 - fw, W / 2 + fw, z0 + H, z0 + H + fw), (-W / 2 - fw, W / 2 + fw, z0 - fw, z0),
                           (-W / 2 - fw, -W / 2, z0, z0 + H), (W / 2, W / 2 + fw, z0, z0 + H)):
        m.rounded_box("alu", ((x0 + x1) / 2, -depth / 2, (za + zb) / 2), ((x1 - x0) / 2, depth / 2, (zb - za) / 2), 0.003,
                      n_arc=1)  # one bevel segment: the same look at a quarter of the faces
    for sx in (-1, 1):
        for zz in (z0 - fw / 2, z0 + H + fw / 2):
            m.rounded_box("corner", (sx * (W / 2 + fw / 2), -depth / 2 - 0.001, zz), (0.014, depth / 2 + 0.001, 0.014), 0.004)
    # the tray: a shallow channel 6 cm deep along the bottom (a floor and a front lip)
    m.box("alu", (0.0, -0.035, 0.004), (W / 2 - 0.05, 0.035, 0.004))
    m.box("alu", (0.0, -0.068, 0.012), (W / 2 - 0.05, 0.002, 0.012))
    for k, (mat, x) in enumerate((("marker_blue", -0.3), ("marker_black", -0.22), ("marker_red", 0.15))):
        m.cylinder(mat, (0.0, 0.0, 0.0), 0.009, 0.06, seg=12)
        v, f, n, _ = m.parts[mat].pop()
        R = np.array([[0, 0, 1.0], [0, 1.0, 0], [-1.0, 0, 0]])  # lying along x
        m.add(mat, v @ R.T + [x, -0.03 - 0.012 * (k % 2), 0.017], f, n @ R.T)
        m.cylinder("marker_cap", (0.0, 0.0, 0.0), 0.0105, 0.0075, seg=12)  # the cap at one end
        v, f, n, _ = m.parts.pop("marker_cap")[0]  # built once, then given the marker's own cap colour
        m.add(mat + "_cap", v @ R.T + [x + 0.0525, -0.03 - 0.012 * (k % 2), 0.017], f, n @ R.T)
    m.rounded_box("eraser", (0.4, -0.035, 0.022), (0.075, 0.025, 0.014), 0.006)
    m.box("felt", (0.4, -0.035, 0.0085), (0.074, 0.024, 0.0005))
    return m.write("proc_whiteboard", "Whiteboard with tray (procedural)", "visual only, on a wall",
                   {"board": whiteboard_texture()})


def storage_carton_top() -> dict:
    """The carton on the corner stack (0.39 x 0.52 x 0.30 m), its own label and marks."""
    m = pf.Mesh()
    L, W, H = 0.52, 0.39, 0.30
    img, rects = carton_texture(L, W, H, 202, label=True, fragile=True)
    carton(m, (0.0, 0.0, 0.0), (L, W, H), 0.0, rects)
    return m.write("proc_storage_carton_top", "Carton (procedural)", "member: one box", {"carton": img})


# ----- building-standard wall and ceiling items (one material and one texture each: one draw call) -----
pf.MATERIALS.update({
    "plate": ((1.0, 1.0, 1.0, 1.0), 0.3, 0.35, 0.0),  # white moulded plastic, textured
    "sign_face": ((1.0, 1.0, 1.0, 1.0), 0.25, 0.3, 0.0),
})
PLATE_PPM = 2000  # texture pixels per metre on small plates (0.5 mm per pixel)


def _plate_image(w_m, h_m, base=(236, 235, 230)):
    """A white plastic face with a soft bevel (darker rim, lighter inner edge)."""
    w, h = max(8, int(w_m * PLATE_PPM)), max(8, int(h_m * PLATE_PPM))
    yy, xx = np.mgrid[0:h, 0:w].astype(float)
    edge = np.minimum(np.minimum(xx, w - 1 - xx), np.minimum(yy, h - 1 - yy))
    k = 1.0 - 0.14 * np.exp(-edge / 4.0) + 0.04 * np.exp(-((edge - 6.0) / 2.0) ** 2)
    img = np.array(base, float)[None, None, :] * k[..., None]
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))


def _wall_plate(model_id, name, size, front: Image.Image, side_rgb=(222, 221, 216)) -> dict:
    """A plate size = (width, height, depth) on a wall: back at y = 0, facing -y, base at z = 0. The
    front face shows `front`; the edges use a strip of plain plastic at the image's right."""
    W, H, D = size
    side = Image.new("RGB", (8, front.height), side_rgb)
    img = Image.new("RGB", (front.width + 8, front.height))
    img.paste(front, (0, 0))
    img.paste(side, (front.width, 0))
    fu = front.width / img.width
    edge = (fu + 0.01, 1.0, 1.0, 0.0)
    m = pf.Mesh()
    m.textured_box("plate", (-W / 2, -D, 0.0), (W / 2, 0.0, H),
                   {"-y": (0.0, 1.0, fu, 0.0), "+y": edge, "-x": edge, "+x": edge, "+z": edge, "-z": edge})
    return m.write(model_id, name, "visual only, on a wall", {"plate": img})


def socket() -> dict:
    """A double socket outlet (two round recessed outlets with two pin holes and earth clips),
    15 x 8 cm, 9 mm proud."""
    W, H = 0.15, 0.08
    img = _plate_image(W, H)
    d = ImageDraw.Draw(img)
    s = PLATE_PPM
    for cx in (0.0375, 0.1125):
        x, y, r = int(cx * s), int(0.04 * s), int(0.019 * s)
        d.ellipse([x - r, y - r, x + r, y + r], fill=(214, 213, 208), outline=(170, 170, 166), width=3)
        for dx in (-0.0095, 0.0095):
            d.ellipse([x + int(dx * s) - 9, y - 9, x + int(dx * s) + 9, y + 9], fill=(40, 40, 40))
        for dy in (-0.016, 0.016):
            d.rectangle([x - 10, y + int(dy * s) - 4, x + 10, y + int(dy * s) + 4], fill=(160, 160, 158))
    return _wall_plate("proc_socket", "Double socket outlet (procedural)", (W, H, 0.009), img)


def light_switch() -> dict:
    """A rocker light switch, 8.5 x 8.5 cm, 10 mm proud: the rocker drawn with its lower half pressed."""
    W = 0.085
    img = _plate_image(W, W)
    d = ImageDraw.Draw(img)
    s = PLATE_PPM
    a, b = int(0.018 * s), int(W * s) - int(0.018 * s)
    d.rectangle([a, a, b, b], fill=(244, 243, 238), outline=(186, 186, 182), width=3)
    mid = (a + b) // 2
    d.rectangle([a + 3, mid, b - 3, b - 3], fill=(226, 225, 220))  # the pressed half, in shade
    d.line([(a + 3, mid), (b - 3, mid)], fill=(170, 170, 166), width=2)
    return _wall_plate("proc_switch", "Light switch (procedural)", (W, W, 0.01), img)


def thermostat() -> dict:
    """A room thermostat, 9 x 9 cm, 25 mm proud: a small grey display and two buttons."""
    W = 0.09
    img = _plate_image(W, W)
    d = ImageDraw.Draw(img)
    s = PLATE_PPM
    d.rectangle([int(0.02 * s), int(0.018 * s), int(0.07 * s), int(0.045 * s)], fill=(150, 162, 150),
                outline=(90, 90, 90), width=3)
    for k, bx in enumerate((0.03, 0.06)):
        d.ellipse([int(bx * s) - 14, int(0.065 * s) - 14, int(bx * s) + 14, int(0.065 * s) + 14],
                  fill=(210, 210, 206), outline=(150, 150, 148), width=2)
    return _wall_plate("proc_thermostat", "Thermostat (procedural)", (W, W, 0.025), img)


def _running_man(d, x, y, s, color):
    """The ISO 7010 emergency exit pictogram, simplified: a running figure, a door, an arrow."""
    c = color
    d.ellipse([x + 0.30 * s, y + 0.05 * s, x + 0.42 * s, y + 0.17 * s], fill=c)  # head
    w = int(0.07 * s)
    for (a, b) in (((0.34, 0.2), (0.27, 0.48)), ((0.27, 0.48), (0.12, 0.62)), ((0.27, 0.48), (0.4, 0.62)),
                   ((0.4, 0.62), (0.36, 0.8)), ((0.32, 0.26), (0.48, 0.36)), ((0.32, 0.26), (0.16, 0.34)),
                   ((0.12, 0.62), (0.02, 0.6))):
        d.line([(x + a[0] * s, y + a[1] * s), (x + b[0] * s, y + b[1] * s)], fill=c, width=w)
    d.rectangle([x + 0.55 * s, y + 0.05 * s, x + 0.85 * s, y + 0.82 * s], outline=c, width=int(0.05 * s))  # the door


def exit_sign() -> dict:
    """An emergency exit sign, 35 x 13 cm, 4 cm deep: a green face with the white running man, a door
    and an arrow, in a white housing."""
    W, H = 0.35, 0.13
    s = PLATE_PPM // 2
    img = Image.new("RGB", (int(W * s), int(H * s)), (236, 236, 232))
    d = ImageDraw.Draw(img)
    g = (18, 140, 72)
    m = int(0.008 * s)
    d.rectangle([m, m, img.width - m, img.height - m], fill=g)
    _running_man(d, int(0.03 * s), int(0.014 * s), int(0.1 * s), (250, 250, 250))
    ax, ay = int(0.2 * s), img.height // 2  # the arrow, pointing along the escape route
    d.rectangle([ax, ay - int(0.012 * s), ax + int(0.08 * s), ay + int(0.012 * s)], fill=(250, 250, 250))
    d.polygon([(ax + int(0.08 * s), ay - int(0.035 * s)), (ax + int(0.13 * s), ay),
               (ax + int(0.08 * s), ay + int(0.035 * s))], fill=(250, 250, 250))
    return _wall_plate("proc_exit_sign", "Emergency exit sign (procedural)", (W, H, 0.04), img)


def first_aid_kit() -> dict:
    """A wall-mounted first aid cabinet, 30 x 25 cm, 10 cm deep: green with a white cross and a latch."""
    W, H = 0.3, 0.25
    s = PLATE_PPM // 2
    img = _plate_image(W, H, base=(20, 128, 70)).resize((int(W * s), int(H * s)))
    d = ImageDraw.Draw(img)
    cx, cy, a, b = img.width // 2, img.height // 2, int(0.03 * s), int(0.09 * s)
    d.rectangle([cx - a, cy - b, cx + a, cy + b], fill=(246, 246, 242))
    d.rectangle([cx - b, cy - a, cx + b, cy + a], fill=(246, 246, 242))
    d.rectangle([img.width - int(0.03 * s), cy - int(0.02 * s), img.width - int(0.018 * s), cy + int(0.02 * s)],
                fill=(30, 90, 55))
    return _wall_plate("proc_first_aid", "First aid cabinet (procedural)", (W, H, 0.1), img, side_rgb=(18, 118, 64))


def notice_board() -> dict:
    """A cork notice board, 90 x 60 cm in a thin aluminium frame, with pinned sheets (lines and blocks
    of print, no words), a photo, coloured pins; 2 cm deep."""
    W, H = 0.9, 0.6
    s = 900
    rng = np.random.default_rng(91)
    w, h = int(W * s), int(H * s)
    cork = np.array([176.0, 132.0, 84.0])[None, None, :] * (
        1.0 + 0.10 * _noise(rng, h, w, 0.4) + 0.05 * _noise(rng, h, w, 1.6))[..., None]
    img = Image.fromarray(np.clip(cork, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(img)
    f = int(0.018 * s)
    d.rectangle([0, 0, w - 1, h - 1], outline=(170, 172, 174), width=f)
    for k in range(6):
        pw, ph = int(rng.uniform(0.14, 0.21) * s), int(rng.uniform(0.18, 0.27) * s)
        x, y = int(rng.uniform(0.04, W - 0.25) * s), int(rng.uniform(0.04, H - 0.3) * s)
        paper = (244, 243, 236) if k % 3 else (250, 238, 160)
        d.rectangle([x + 3, y + 3, x + pw + 3, y + ph + 3], fill=(120, 90, 60))  # its shadow on the cork
        d.rectangle([x, y, x + pw, y + ph], fill=paper)
        if k == 4:  # a photo
            d.rectangle([x + 8, y + 8, x + pw - 8, y + ph // 2], fill=(110, 140, 170))
            d.rectangle([x + 8, y + ph // 4, x + pw - 8, y + ph // 2], fill=(90, 120, 80))
        d.rectangle([x + 10, y + 12, x + int(pw * 0.7), y + 20], fill=(60, 60, 64))  # a heading block
        for ly in range(y + 34, y + ph - 12, 12):
            d.rectangle([x + 10, ly, x + int(rng.uniform(0.4, 0.92) * pw), ly + 3], fill=(140, 140, 146))
        pin = [(200, 40, 40), (40, 90, 190), (40, 150, 60), (230, 190, 30)][k % 4]
        d.ellipse([x + pw // 2 - 7, y + 4, x + pw // 2 + 7, y + 18], fill=pin)
    return _wall_plate("proc_notice_board", "Cork notice board (procedural)", (W, H, 0.02), img,
                       side_rgb=(170, 172, 174))


def safety_poster() -> dict:
    """A framed A2 safety poster (42 x 59 cm, landscape): a blue header band, four pictogram panels
    (mandatory blue circles and a warning triangle, abstract figures), blocks of print, no words."""
    W, H = 0.594, 0.42
    s = 900
    w, h = int(W * s), int(H * s)
    img = Image.new("RGB", (w, h), (248, 248, 244))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, w - 1, h - 1], outline=(40, 40, 44), width=int(0.012 * s))
    d.rectangle([int(0.02 * s), int(0.02 * s), w - int(0.02 * s), int(0.08 * s)], fill=(20, 80, 160))
    for k in range(4):
        cx, cy, r = int((0.09 + k * 0.14) * s), int(0.19 * s), int(0.05 * s)
        if k == 3:
            d.polygon([(cx, cy - r), (cx + r, cy + r), (cx - r, cy + r)], fill=(250, 200, 20), outline=(20, 20, 20))
            d.rectangle([cx - 3, cy - r // 3, cx + 3, cy + r // 3], fill=(20, 20, 20))
        else:
            d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(20, 80, 160))
            d.ellipse([cx - r // 4, cy - r // 2, cx + r // 4, cy - r // 6], fill=(250, 250, 250))
            d.rectangle([cx - r // 5, cy - r // 6, cx + r // 5, cy + r // 2], fill=(250, 250, 250))
        for ly in range(int(0.27 * s), int(0.39 * s), 10):
            d.rectangle([cx - r, ly, cx + r - (ly % 3) * 6, ly + 3], fill=(120, 120, 126))
    return _wall_plate("proc_safety_poster", "Framed safety poster (procedural)", (W, H, 0.012), img,
                       side_rgb=(40, 40, 44))


def trunking() -> dict:
    """White PVC dado trunking, 100 x 50 mm, 1.2 m long, with its lid seam and end caps."""
    L, Hh, D = 1.2, 0.1, 0.05
    img = _plate_image(L / 3, Hh)  # tiled 3 times along the length
    d = ImageDraw.Draw(img)
    d.line([(0, img.height // 2), (img.width, img.height // 2)], fill=(200, 200, 196), width=2)  # the lid seam
    m = pf.Mesh()
    side = (0.0, 0.0, 0.05, 0.05)
    m.textured_box("plate", (-L / 2, -D, 0.0), (L / 2, 0.0, Hh),
                   {"-y": (0.0, 1.0, 3.0, 0.0), "+y": side, "-x": side, "+x": side, "+z": side, "-z": side})
    return m.write("proc_trunking", "Dado trunking (procedural)", "visual only, on a wall", {"plate": img})


def smoke_detector() -> dict:
    """A ceiling smoke detector: an 11 cm white dome 3.5 cm deep, a vent ring and a red LED on its
    face. Its top (the ceiling side) at z = 0.035: base at z = 0 is the face."""
    m = pf.Mesh()
    m.cylinder("plate", (0.0, 0.0, 0.0175), 0.055, 0.0175, seg=32, radius_top=0.05)
    v, f, n, _ = m.parts["plate"].pop()
    face = Image.new("RGB", (256, 256), (238, 238, 234))
    dd = ImageDraw.Draw(face)
    for r in range(70, 118, 9):  # the vent slots
        dd.ellipse([128 - r, 128 - r, 128 + r, 128 + r], outline=(170, 170, 168), width=3)
    dd.ellipse([150, 150, 162, 162], fill=(200, 30, 30))  # the LED
    uv = np.column_stack([(v[:, 0] / 0.055 + 1) / 2, (v[:, 1] / 0.055 + 1) / 2])  # planar from below
    m.add("plate", v, f, n, uv)
    return m.write("proc_smoke_detector", "Smoke detector (procedural)", "visual only, on the ceiling",
                   {"plate": face})


# ----- framed prints, the TV, the extinguisher, the dome camera -----
pf.MATERIALS.update({
    "frame_oak": ((1.0, 1.0, 1.0, 1.0), 0.25, 0.3, 0.0),
    "print": ((1.0, 1.0, 1.0, 1.0), 0.15, 0.2, 0.0),
    "tv_body": ((0.05, 0.05, 0.055, 1.0), 0.4, 0.5, 0.0),
    "ext_red": ((0.72, 0.06, 0.05, 1.0), 0.6, 0.7, 0.0),
    "ext_black": ((0.05, 0.05, 0.05, 1.0), 0.4, 0.45, 0.0),
    "ext_steel": ((0.62, 0.63, 0.64, 1.0), 0.8, 0.8, 0.05),
    "ext_label": ((1.0, 1.0, 1.0, 1.0), 0.2, 0.3, 0.0),
    "dome": ((0.12, 0.12, 0.13, 1.0), 0.8, 0.9, 0.0),
})


def landscape(w, h, seed):
    """A landscape print: a graded sky with soft clouds, three ridges fading with distance (aerial
    perspective), a lake reflecting the sky, a near shore; painted look (slightly blurred)."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(float)
    t = yy / h
    sky_top, sky_low = np.array([92.0, 140, 196]), np.array([214.0, 222, 226])
    img = sky_top[None, None] * (1 - t[..., None]) + sky_low[None, None] * t[..., None]
    cloud = np.clip(_noise(rng, h, w, 3.2, stretch=1 / 3.0) - 0.4, 0, None)[..., None]  # long horizontal clouds
    img = img + 40 * cloud * (t[..., None] < 0.5)
    horizon = int(0.62 * h)
    cols = [np.array([120.0, 146, 170]), np.array([84.0, 112, 106]), np.array([52.0, 80, 58])]
    for k, col in enumerate(cols):  # far to near
        base = horizon - (0.30 - 0.09 * k) * h
        ridge = base + 0.07 * h * _noise(rng, 1, w, 2.4 + 0.3 * k)[0] * (1 + 0.3 * k)
        mask = yy > ridge[None, :]
        img[mask] = (img[mask] * 0.15 + col * 0.85) * (1.0 + 0.04 * _noise(rng, h, w, 1.2)[mask])[:, None]
    water = yy > horizon
    flip = np.clip(2 * horizon - yy, 0, h - 1).astype(int)
    refl = img[flip, xx.astype(int)]
    img[water] = refl[water] * 0.7 + np.array([60.0, 90, 110]) * 0.3
    shore = yy > horizon + (0.22 + 0.04 * _noise(rng, 1, w, 2.0)[0])[None, :] * h
    img[shore] = np.array([70.0, 92, 52]) * (1.0 + 0.08 * _noise(rng, h, w, 1.0)[shore])[:, None]
    out = Image.fromarray(np.clip(img * 1.12, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1.2))
    return out


def framed_print(model_id, art_w, art_h, seed) -> dict:
    """A framed landscape print: an oak moulding 3 cm wide and 2.5 cm deep, a white mat 6 cm wide
    around the print, the print 3 mm behind the mat; back at y = 0, facing -y, base at z = 0."""
    fw, mat, depth = 0.03, 0.06, 0.025
    W, H = art_w + 2 * (mat + fw), art_h + 2 * (mat + fw)
    s = 700
    img = Image.new("RGB", (int(W * s), int(H * s)), (242, 240, 232))  # the mat
    art = landscape(int(art_w * s), int(art_h * s), seed)
    img.paste(art, (int((fw + mat) * s), int((fw + mat) * s)))
    d = ImageDraw.Draw(img)
    a = int((fw + mat) * s)
    d.rectangle([a - 2, a - 2, a + art.width + 1, a + art.height + 1], outline=(200, 198, 190), width=2)  # bevel
    m = pf.Mesh()
    m.textured_box("print", (-W / 2 + fw, -0.012, fw), (W / 2 - fw, 0.0, H - fw),
                   {"-y": (fw / W, 1 - fw / H, 1 - fw / W, fw / H), "+y": (0, 0, 0.01, 0.01), "-x": (0, 0, 0.01, 0.01),
                    "+x": (0, 0, 0.01, 0.01), "+z": (0, 0, 0.01, 0.01), "-z": (0, 0, 0.01, 0.01)})
    oak = pf.surface_texture("oak", (0.55, 0.4, 0.25, 1.0), seed)
    for x0, x1, z0, z1 in ((-W / 2, W / 2, H - fw, H), (-W / 2, W / 2, 0.0, fw), (-W / 2, -W / 2 + fw, fw, H - fw),
                           (W / 2 - fw, W / 2, fw, H - fw)):
        m.rounded_box("frame_oak", ((x0 + x1) / 2, -depth / 2, (z0 + z1) / 2), ((x1 - x0) / 2, depth / 2, (z1 - z0) / 2),
                      0.004, n_arc=1)
    return m.write(model_id, "Framed print (procedural)", "visual only, on a wall", {"print": img, "frame_oak": oak})


def print_corridor_a() -> dict:
    return framed_print("proc_print_a", 0.62, 0.38, 11)


def print_corridor_b() -> dict:
    return framed_print("proc_print_b", 0.62, 0.38, 12)


def print_reception() -> dict:
    return framed_print("proc_print_c", 0.52, 0.34, 13)


def television() -> dict:
    """A 47-inch television, switched off: 1.04 x 0.6 m panel, 9 mm bezel, 3 cm deep with a back box,
    the screen's dark glass and the room's reflection baked in, a standby light and a logo strip;
    on a flat wall bracket. Back at y = 0, facing -y, base at z = 0."""
    W, H, D = 1.04, 0.6, 0.03
    m = pf.Mesh()
    m.rounded_box("tv_body", (0.0, -0.02 - D / 2, H / 2), (W / 2, D / 2, H / 2), 0.004, n_arc=1)
    m.rounded_box("tv_body", (0.0, -0.02 + 0.006, H / 2 - 0.03), (0.3, 0.014, 0.18), 0.01, n_arc=1)  # back box
    m.box("ext_steel", (0.0, -0.004, H / 2 - 0.03), (0.2, 0.004, 0.015))  # the bracket's rail
    bez = 0.009
    q = np.array([(-W / 2 + bez, -0.02 - D - 0.0004, H - bez), (W / 2 - bez, -0.02 - D - 0.0004, H - bez),
                  (W / 2 - bez, -0.02 - D - 0.0004, bez + 0.006), (-W / 2 + bez, -0.02 - D - 0.0004, bez + 0.006)])
    m.add("screen", q, [(0, 2, 1), (0, 3, 2)], np.tile([0.0, -1.0, 0.0], (4, 1)),
          np.array([(0.0, 1.0), (1.0, 1.0), (1.0, 0.0), (0.0, 0.0)]))
    m.box("monitor_logo", (0.0, -0.02 - D - 0.0003, 0.004), (0.03, 0.0004, 0.002))
    return m.write("proc_tv", "Television (procedural)", "visual only, on a wall",
                   {"screen": screen_texture(1040, 600)})


def extinguisher() -> dict:
    """A 6 kg powder extinguisher on its wall bracket: red body 16 cm across and 47 cm tall with a
    domed shoulder, a white instruction band (blocks, no words), a black valve and carrying handle,
    a gauge, a hose looping down; total about 0.58 m. Back (the bracket) at y = 0, facing -y."""
    m = pf.Mesh()
    r, cy = 0.08, -0.095
    m.cylinder("ext_red", (0.0, cy, 0.215), r, 0.215, seg=32)
    m.cushion("ext_red", (0.0, cy, 0.43), (r, r, 0.045), e=0.9, nu=32, nv=10)
    band = Image.new("RGB", (256, 128), (238, 238, 232))
    d = ImageDraw.Draw(band)
    d.rectangle([0, 0, 255, 14], fill=(200, 30, 25))
    for k in range(5):
        d.rectangle([90, 26 + 18 * k, 90 + 100 - 12 * k, 32 + 18 * k], fill=(90, 90, 96))
    d.ellipse([20, 30, 70, 80], outline=(200, 30, 25), width=5)
    m.cylinder("ext_label", (0.0, cy, 0.23), r + 0.0008, 0.09, seg=32, caps=False, tile=None)
    v, f, n, _ = m.parts["ext_label"].pop()
    ang = np.arctan2(v[:, 0], -(v[:, 1] - cy))  # the label's middle faces the room (-y)
    uv = np.column_stack([(ang / (2 * math.pi) + 0.5) * 2.0, (v[:, 2] - 0.14) / 0.18])
    m.add("ext_label", v, f, n, uv)
    m.cylinder("ext_black", (0.0, cy, 0.49), 0.02, 0.02, seg=16)  # valve
    m.box("ext_black", (0.0, cy - 0.04, 0.52), (0.012, 0.05, 0.006), pitch_deg=8.0)  # handle and lever
    m.box("ext_black", (0.0, cy - 0.03, 0.495), (0.011, 0.045, 0.005), pitch_deg=-6.0)
    m.cylinder("ext_steel", (0.0, cy - 0.025, 0.49), 0.012, 0.004, seg=16)  # gauge
    hose = np.array([(0.025, cy, 0.48), (0.07, cy - 0.01, 0.42), (0.095, cy - 0.01, 0.3), (0.09, cy - 0.015, 0.16),
                     (0.07, cy - 0.04, 0.1)])
    for a, b in zip(hose, hose[1:]):
        mid = (a + b) / 2
        m.tube("ext_black", np.array([a, b]) - mid, np.tile([1.0, 0.0, 0.0], (2, 1)), 0.008, mid)
    m.box("ext_steel", (0.0, -0.008, 0.38), (0.035, 0.008, 0.05))  # the bracket
    return m.write("proc_extinguisher", "Fire extinguisher on a bracket (procedural)", "visual only, on a wall",
                   {"ext_label": band})


def fire_sign() -> dict:
    """The fire extinguisher sign: a red square with a white extinguisher pictogram, 20 x 20 cm."""
    img = Image.new("RGB", (200, 200), (196, 26, 22))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([82, 56, 118, 170], radius=12, fill=(250, 250, 250))
    d.rectangle([92, 38, 108, 58], fill=(250, 250, 250))
    d.line([(108, 46), (140, 40), (150, 80)], fill=(250, 250, 250), width=6)
    return _wall_plate("proc_fire_sign", "Fire extinguisher sign (procedural)", (0.2, 0.2, 0.003), img)


def call_point() -> dict:
    """A manual fire alarm call point: red, 9 x 9 cm, 5 cm proud, a white glass window."""
    img = _plate_image(0.09, 0.09, base=(200, 30, 26))
    d = ImageDraw.Draw(img)
    a, b = int(0.02 * PLATE_PPM), int(0.07 * PLATE_PPM)
    d.rectangle([a, a, b, b], fill=(236, 236, 232), outline=(150, 20, 18), width=4)
    d.polygon([(a + 20, b - 20), ((a + b) // 2, a + 25), (b - 20, b - 20)], outline=(60, 60, 60), width=5)
    return _wall_plate("proc_call_point", "Fire alarm call point (procedural)", (0.09, 0.09, 0.05), img,
                       side_rgb=(190, 28, 24))


def dome_camera() -> dict:
    """A ceiling security camera: a white base ring and a dark smoked dome, 12 cm across; face at z = 0."""
    m = pf.Mesh()
    m.cylinder("plate", (0.0, 0.0, 0.03), 0.06, 0.008, seg=32)
    m.cushion("dome", (0.0, 0.0, 0.022), (0.045, 0.045, 0.022), e=1.0, nu=32, nv=12)
    face = Image.new("RGB", (64, 64), (236, 236, 232))
    v, f, n, _ = m.parts["plate"].pop()
    m.add("plate", v, f, n, np.column_stack([(v[:, 0] / 0.06 + 1) / 2, (v[:, 1] / 0.06 + 1) / 2]))
    return m.write("proc_dome_camera", "Dome camera (procedural)", "visual only, on the ceiling", {"plate": face})


# ----- lab bench props -----
pf.MATERIALS.update({
    "lab_body": ((0.9, 0.9, 0.88, 1.0), 0.35, 0.4, 0.0),  # instrument white enamel
    "lab_black": ((0.07, 0.07, 0.08, 1.0), 0.4, 0.5, 0.0),
    "lab_chrome": ((0.7, 0.71, 0.73, 1.0), 0.85, 0.85, 0.08),
    "glass": ((0.82, 0.9, 0.95, 0.32), 0.95, 0.95, 0.0),  # thin clear glass (alpha)
    "liquid_blue": ((0.2, 0.45, 0.85, 0.7), 0.6, 0.8, 0.0),
    "liquid_amber": ((0.85, 0.55, 0.15, 0.7), 0.6, 0.8, 0.0),
    "rack": ((0.85, 0.85, 0.82, 1.0), 0.3, 0.35, 0.0),
    "analyzer_face": ((1.0, 1.0, 1.0, 1.0), 0.4, 0.5, 0.0),
    "glove_box": ((1.0, 1.0, 1.0, 1.0), 0.15, 0.2, 0.0),
    "glove_blue": ((0.25, 0.45, 0.85, 1.0), 0.25, 0.35, 0.0),
})


def _rot3(v, n, yaw=0.0, pitch=0.0, roll=0.0):
    """Rotate points and normals: roll about y, then pitch about x, then yaw about z."""
    r, p, y = (math.radians(a) for a in (roll, pitch, yaw))
    Ry = np.array([[math.cos(r), 0, math.sin(r)], [0, 1, 0], [-math.sin(r), 0, math.cos(r)]])
    Rx = np.array([[1, 0, 0], [0, math.cos(p), -math.sin(p)], [0, math.sin(p), math.cos(p)]])
    Rz = np.array([[math.cos(y), -math.sin(y), 0], [math.sin(y), math.cos(y), 0], [0, 0, 1]])
    R = Rz @ Rx @ Ry
    return v @ R.T, n @ R.T


def _place(m, mat, build, at, yaw=0.0, pitch=0.0, roll=0.0):
    """Build a part at the origin with `build(m)` under a temporary material, then rotate and move it."""
    build(m)
    v, f, n, uv = m.parts["_tmp"].pop()
    del m.parts["_tmp"]
    v, n = _rot3(v, n, yaw, pitch, roll)
    m.add(mat, v + np.asarray(at, float), f, n, uv)


def _beaker(m, at, r, h, liquid=None, fill=0.5):
    """An open glass beaker (wall in and out, a base, a rim), optionally with liquid."""
    x, y = at[0], at[1]
    m.cylinder("glass", (x, y, h / 2), r, h / 2, seg=24, caps=False)
    m.cylinder("glass", (x, y, h / 2), r - 0.0015, h / 2 - 0.001, seg=24, caps=False, inward=True)
    m.annulus("glass", 0.0, 0.0, r, seg=24, up=False)
    v, f, n, uv = m.parts["glass"].pop()
    m.add("glass", v + [x, y, 0.0], f, n, uv)
    if liquid:
        m.cylinder(liquid, (x, y, 0.002 + fill * h / 2), r - 0.002, fill * h / 2, seg=24)


def analyzer_screen(w=320, h=180):
    """The analyzer's display: a dark panel with a plotted trace, a few status blocks, no words."""
    img = Image.new("RGB", (w, h), (18, 30, 40))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, w, 18], fill=(30, 70, 110))
    pts = [(10 + k * 3, 120 - 40 * math.exp(-((k - 50) / 9.0) ** 2) - 12 * math.exp(-((k - 22) / 5.0) ** 2))
           for k in range(95)]
    d.line(pts, fill=(90, 220, 140), width=2)
    d.line([(10, 125), (300, 125)], fill=(80, 100, 110), width=1)
    for k in range(3):
        d.rectangle([220, 40 + 22 * k, 300, 54 + 22 * k], fill=(40, 60, 76))
        d.rectangle([224, 44 + 22 * k, 224 + 20 + 18 * k, 50 + 22 * k], fill=(150, 170, 180))
    return img


def lab_bench_set() -> dict:
    """On the long lab bench (frame: the worktop's centre, z = 0 on the top, front at y = -0.42): a
    microscope, glassware with liquids, a test tube rack, and a benchtop analyzer with a display."""
    m = pf.Mesh()
    # microscope (at x -0.6): base, C-arm, stage, objective turret, two eyepieces, focus knobs
    mx, my = -0.6, 0.08
    m.rounded_box("lab_body", (mx, my, 0.015), (0.1, 0.13, 0.015), 0.01, n_arc=1)
    m.rounded_box("lab_body", (mx, my + 0.1, 0.14), (0.03, 0.025, 0.12), 0.01, n_arc=1)  # the pillar
    m.rounded_box("lab_body", (mx, my + 0.04, 0.27), (0.035, 0.085, 0.024), 0.008, n_arc=1)  # the head, over the stage
    m.box("lab_black", (mx, my - 0.01, 0.1), (0.07, 0.065, 0.005))  # the stage
    m.box("lab_body", (mx, my + 0.065, 0.0925), (0.02, 0.015, 0.0075))  # its bracket on the pillar
    m.cylinder("lab_chrome", (mx, my - 0.01, 0.032), 0.018, 0.002, seg=20)  # the light under the stage
    m.cylinder("lab_body", (mx, my - 0.01, 0.21), 0.015, 0.04, seg=16)  # the nosepiece
    m.cylinder("lab_chrome", (mx, my - 0.01, 0.16), 0.022, 0.012, seg=20)  # the turret
    for k in range(3):  # objectives on a ring, tilted out
        a = math.radians(90 + 120 * k)
        _place(m, "lab_chrome", lambda q: q.cylinder("_tmp", (0, 0, -0.015), 0.005, 0.015, seg=10),
               (mx + 0.013 * math.cos(a), my - 0.01 + 0.013 * math.sin(a), 0.152), yaw=math.degrees(a) - 90,
               pitch=-10.0)
    for dx in (-0.018, 0.018):
        _place(m, "lab_black", lambda q: q.cylinder("_tmp", (0, 0, 0), 0.009, 0.03, seg=14),
               (mx + dx, my + 0.0, 0.31), pitch=35.0)  # eyepieces on the head, tilted to the user
    for sx in (-1, 1):
        _place(m, "lab_black", lambda q: q.cylinder("_tmp", (0, 0, 0), 0.02, 0.008, seg=18),
               (mx + sx * 0.037, my + 0.1, 0.09), roll=90.0)  # focus knobs, against the pillar
    # glassware (x -0.25 to 0.05): two beakers (one with blue liquid), a flask with amber liquid,
    # a graduated cylinder
    _beaker(m, (-0.25, -0.1), 0.035, 0.09, "liquid_blue", 0.45)
    _beaker(m, (-0.16, -0.05), 0.028, 0.07)
    m.cylinder("glass", (-0.06, -0.08, 0.04), 0.045, 0.04, seg=24, radius_top=0.016)  # flask body
    m.cylinder("glass", (-0.06, -0.08, 0.1), 0.014, 0.02, seg=16)  # its neck
    m.cylinder("liquid_amber", (-0.06, -0.08, 0.02), 0.04, 0.018, seg=24, radius_top=0.03)
    m.cylinder("glass", (0.02, 0.02, 0.006), 0.03, 0.006, seg=20)  # graduated cylinder: foot on the top
    m.cylinder("glass", (0.02, 0.02, 0.125), 0.013, 0.115, seg=16, caps=False)
    # test tube rack with six tubes (x 0.2)
    rx, ry = 0.22, -0.02
    m.box("rack", (rx, ry, 0.05), (0.09, 0.03, 0.004))
    m.box("rack", (rx, ry, 0.002), (0.09, 0.03, 0.002))
    for sx in (-1, 1):
        m.box("rack", (rx + sx * 0.087, ry, 0.026), (0.003, 0.03, 0.026))
    for k in range(6):
        tx = rx - 0.07 + k * 0.028
        m.cylinder("glass", (tx, ry, 0.06), 0.007, 0.055, seg=12)
        if k % 2 == 0:
            m.cylinder("liquid_blue" if k != 4 else "liquid_amber", (tx, ry, 0.025), 0.0062, 0.018, seg=12)
    # benchtop analyzer (x 0.7): a rounded white case, a display, buttons, a sample drawer
    ax, ay = 0.7, 0.05
    m.rounded_box("lab_body", (ax, ay, 0.1), (0.2, 0.17, 0.1), 0.012, n_arc=1)
    q = np.array([(ax - 0.12, ay - 0.1705, 0.17), (ax + 0.04, ay - 0.1705, 0.17), (ax + 0.04, ay - 0.1705, 0.08),
                  (ax - 0.12, ay - 0.1705, 0.08)])
    m.add("analyzer_face", q, [(0, 2, 1), (0, 3, 2)], np.tile([0.0, -1.0, 0.0], (4, 1)),
          np.array([(0.0, 1.0), (1.0, 1.0), (1.0, 0.0), (0.0, 0.0)]))
    for k in range(3):
        m.cylinder("lab_black", (ax + 0.09 + 0.03 * (k % 2), ay - 0.171, 0.15 - 0.03 * (k // 2)), 0.008, 0.002,
                   seg=12)
        v, f, n, uv = m.parts["lab_black"].pop()
        v2, n2 = _rot3(v - [ax + 0.09 + 0.03 * (k % 2), ay - 0.171, 0.15 - 0.03 * (k // 2)], n, pitch=90.0)
        m.add("lab_black", v2 + [ax + 0.09 + 0.03 * (k % 2), ay - 0.171, 0.15 - 0.03 * (k // 2)], f, n2, uv)
    m.box("lab_black", (ax, ay - 0.171, 0.035), (0.12, 0.002, 0.012))  # the sample drawer's slot
    return m.write("proc_lab_bench_set", "Lab bench props (procedural)", "visual only, on the lab worktop",
                   {"analyzer_face": analyzer_screen()})


def lab_island_set() -> dict:
    """On the island bench (worktop frame, z = 0 on the top): a box of gloves and a lab notebook."""
    m = pf.Mesh()
    gx, gy = -0.3, -0.05
    # a box of nitrile gloves (24 x 12 x 9 cm, printed carton) with a blue cuff out of its slot
    img = Image.new("RGB", (480, 360), (236, 238, 240))  # rows 0-239: the sides; 240-359: the top
    d = ImageDraw.Draw(img)
    d.ellipse([170, 277, 310, 323], outline=(170, 175, 182), width=3)  # the perforated tear ring
    d.ellipse([180, 286, 300, 314], fill=(28, 30, 34))  # the open slot
    d.rectangle([0, 0, 479, 70], fill=(30, 90, 170))  # the brand band
    d.rectangle([20, 100, 140, 210], fill=(60, 120, 200))  # a glove pictogram block
    d.polygon([(40, 205), (40, 140), (55, 110), (70, 140), (85, 105), (100, 140), (115, 115), (125, 205)],
              fill=(240, 244, 248))
    for k in range(4):  # print lines, no words
        d.rectangle([170, 110 + 24 * k, 170 + 260 - 40 * k, 120 + 24 * k], fill=(120, 130, 145))
    side = (0.0, 1.0, 1.0, 1.0 / 3.0)  # OpenGL v: the image's top two thirds
    lid = (0.0, 1.0 / 3.0, 1.0, 0.0)  # its bottom third
    m.textured_box("glove_box", (gx - 0.12, gy - 0.06, 0.0), (gx + 0.12, gy + 0.06, 0.09),
                   {"-y": side, "+y": side, "-x": (0.0, 1.0, 0.5, 1.0 / 3.0), "+x": (0.0, 1.0, 0.5, 1.0 / 3.0),
                    "+z": lid, "-z": lid})
    for k, (dy, tilt) in enumerate(((-0.004, 35.0), (0.005, 50.0))):  # two layers of a glove's cuff
        _place(m, "glove_blue", lambda q: q.cushion("_tmp", (0, 0, 0), (0.032, 0.004, 0.022), e=0.8, nu=16, nv=8),
               (gx + 0.006 * k, gy + dy, 0.098 + 0.004 * k), pitch=tilt)
    c, s_ = math.cos(math.radians(-6)), math.sin(math.radians(-6))
    rot = np.array([[c, -s_, 0], [s_, c, 0], [0, 0, 1.0]])
    side = (0.0, 0.0, 0.02, 0.02)
    m.textured_box("paper", (0.1, -0.12, 0.0), (0.31, 0.15, 0.012),
                   {"+z": (0.0, 1.0, 1.0, 0.0), "-z": side, "-x": side, "+x": side, "-y": side, "+y": side},
                   rot=rot, pivot=(0.2, 0.0, 0.0))
    return m.write("proc_lab_island_set", "Lab island props (procedural)", "visual only, on the lab worktop",
                   {"paper": notepad_texture(), "glove_box": img})


# ----- what stands on the steel racks: cartons and plastic totes -----
pf.MATERIALS.update({
    "tote_blue": ((0.18, 0.42, 0.78, 1.0), 0.35, 0.4, 0.0),
    "tote_hold": ((0.04, 0.04, 0.05, 1.0), 0.1, 0.1, 0.0),
    "tote_grey": ((0.42, 0.44, 0.46, 1.0), 0.35, 0.4, 0.0),
})
SHELF_TOPS = (0.16, 0.61, 1.06, 1.51)  # deck surfaces of the 2.0 m, 4-level racks (build_world shelving_unit)


def _tote(m, mat, cx, cy, z0, L, W, H):
    """A stacking plastic tote: straight walls with a rim all round, a rib band 4 cm under the rim,
    three vertical ribs on each long side, a dark hand hold on each end, a floor; open top."""
    t = 0.006
    m.box(mat, (cx, cy, z0 + 0.003), (L / 2 - 0.01, W / 2 - 0.01, 0.003))  # floor
    for sx, sy, hx, hy in ((0, -1, L / 2, t), (0, 1, L / 2, t), (-1, 0, t, W / 2), (1, 0, t, W / 2)):
        m.box(mat, (cx + sx * (L / 2 - t), cy + sy * (W / 2 - t), z0 + H / 2), (hx, hy, H / 2))
    for sy in (-1, 1):  # rim and rib band on the long sides, vertical ribs
        m.box(mat, (cx, cy + sy * (W / 2 + 0.003), z0 + H - 0.008), (L / 2 + 0.003, 0.003, 0.008))
        m.box(mat, (cx, cy + sy * (W / 2 + 0.002), z0 + H - 0.045), (L / 2, 0.002, 0.004))
        for k in (-1, 0, 1):
            m.box(mat, (cx + k * L / 4, cy + sy * (W / 2 + 0.002), z0 + H / 2 - 0.02), (0.004, 0.002, H / 2 - 0.03))
    for sx in (-1, 1):  # rim and hand hold on the ends
        m.box(mat, (cx + sx * (L / 2 + 0.003), cy, z0 + H - 0.008), (0.003, W / 2 + 0.003, 0.008))
        m.box("tote_hold", (cx + sx * (L / 2 + 0.0005), cy, z0 + H - 0.035), (0.0012, 0.05, 0.012))


# the racks with cartons (as build_world.build_floor places shelving_unit: centre, half size, seed)
RACKS = (((4.7, 1.9), (0.25, 0.7), 1), ((-4.75, -3.0), (0.25, 1.2), 2), ((-2.9, -3.6), (0.25, 1.2), 3),
         ((-1.15, -3.5), (0.25, 1.2), 4))


def shelf_loads() -> dict:
    """Cartons and totes on every steel rack's four decks, in world coordinates (one model, one
    carton atlas shared by all racks): packed along each rack with gaps, within the deck's depth and
    the clear height to the deck above, some cartons stacked two high."""
    sizes = [(0.3, 0.3, 0.22), (0.4, 0.3, 0.25), (0.25, 0.25, 0.18), (0.35, 0.3, 0.3), (0.2, 0.3, 0.15),
             (0.3, 0.35, 0.28)]
    tiles = [carton_texture(L, min(W, 0.42), H, 300 + k, label=(k % 2 == 0), fragile=(k == 3))
             for k, (L, W, H) in enumerate(sizes)]
    img, rects = atlas(tiles, 3, scale=0.9)
    m = pf.Mesh()
    for (cx, cy), (hx, hy), seed in RACKS:
        rng = np.random.default_rng(seed)
        along_y = hy >= hx
        span, depth = max(hx, hy) - 0.04, min(hx, hy) - 0.03
        rack = pf.Mesh()
        for k, z0 in enumerate(SHELF_TOPS):
            clear = (SHELF_TOPS[k + 1] - z0 - 0.04) if k + 1 < len(SHELF_TOPS) else 0.4
            x = -span + rng.uniform(0.0, 0.05)
            while True:
                if rng.random() < 0.25:  # a plastic tote
                    L, W, H = 0.4, min(0.3, 2 * depth - 0.02), 0.17
                    if x + L > span:
                        break
                    _tote(rack, "tote_blue" if rng.random() < 0.6 else "tote_grey", x + L / 2,
                          rng.uniform(-0.01, 0.01), z0 + 0.0005, L, W, H)
                else:
                    j = int(rng.integers(len(sizes)))
                    L, W, H = sizes[j]
                    W = min(W, 2 * depth - 0.02)
                    if H > clear:
                        j = 4
                        L, W, H = sizes[j]
                        W = min(W, 2 * depth - 0.02)
                    if x + L > span:
                        break
                    y = rng.uniform(-(depth - W / 2) * 0.3, (depth - W / 2) * 0.3)
                    carton(rack, (x + L / 2, y, z0 + 0.0005), (L, W, H), rng.uniform(-3.0, 3.0), rects[j])
                    if H * 2 + 0.01 < clear and rng.random() < 0.3:  # a second one on top, turned a little
                        carton(rack, (x + L / 2 + rng.uniform(-0.01, 0.01), y, z0 + H + 0.001), (L, W, H),
                               rng.uniform(-5.0, 5.0), rects[(j + 2) % len(sizes)])
                x += L + rng.uniform(0.02, 0.09)
        c, s_ = (0.0, 1.0) if along_y else (1.0, 0.0)  # the rack's long axis along world y: turn 90 degrees
        R = np.array([[c, -s_, 0.0], [s_, c, 0.0], [0.0, 0.0, 1.0]])
        for mat, chunks in rack.parts.items():
            for v, f, n, uv in chunks:
                m.add(mat, v @ R.T + [cx, cy, 0.0], f, n @ R.T, uv)
    return m.write("proc_shelf_loads", "Cartons and totes on the racks (procedural)",
                   "visual only, on the rack decks (world coordinates)", {"carton": img}, base_at_zero=False)

# ----- the corridor water cooler -----
pf.MATERIALS.update({
    "cooler_white": ((0.92, 0.92, 0.9, 1.0), 0.35, 0.45, 0.0),
    "cooler_grey": ((0.3, 0.31, 0.33, 1.0), 0.3, 0.35, 0.0),
    "cooler_cold": ((0.15, 0.4, 0.8, 1.0), 0.5, 0.6, 0.0),
    "cooler_hot": ((0.8, 0.15, 0.12, 1.0), 0.5, 0.6, 0.0),
    "bottle_blue": ((0.55, 0.72, 0.92, 0.38), 0.95, 0.95, 0.0),  # tinted clear plastic (alpha)
    "cooler_water": ((0.35, 0.6, 0.9, 0.3), 0.6, 0.8, 0.0),
})


def water_cooler() -> dict:
    """A top-loading water cooler: a white cabinet 31 cm square and 94 cm tall on a dark plinth, a
    dark dispensing panel with a cold and a hot tap over a drip tray, and an upturned 19 litre blue
    bottle (neck in the collar, ribbed, three quarters full); 1.38 m in all. Front faces -y."""
    m = pf.Mesh()
    m.box("cooler_grey", (0.0, 0.0, 0.015), (0.145, 0.145, 0.015))  # the plinth
    m.rounded_box("cooler_white", (0.0, 0.0, 0.485), (0.155, 0.155, 0.455), 0.015, n_arc=2)
    m.box("cooler_grey", (0.0, -0.1555, 0.72), (0.1, 0.002, 0.1))  # the dispensing panel
    for x, mat in ((-0.045, "cooler_cold"), (0.045, "cooler_hot")):
        m.box(mat, (x, -0.17, 0.785), (0.016, 0.014, 0.022))  # the tap paddles
        m.cylinder("cooler_grey", (x, -0.168, 0.755), 0.006, 0.01, seg=10)  # their spouts
    m.box("cooler_grey", (0.0, -0.18, 0.632), (0.09, 0.025, 0.008))  # the drip tray
    m.box("cooler_white", (0.0, -0.18, 0.641), (0.08, 0.02, 0.0015))  # its grille
    m.cylinder("cooler_grey", (0.0, 0.0, 0.955), 0.075, 0.015, seg=28)  # the bottle collar
    m.cylinder("bottle_blue", (0.0, 0.0, 0.99), 0.03, 0.02, seg=20)  # the neck, upside down
    m.cylinder("bottle_blue", (0.0, 0.0, 1.05), 0.03, 0.04, seg=28, radius_top=0.135)  # the shoulder
    m.cylinder("bottle_blue", (0.0, 0.0, 1.215), 0.135, 0.125, seg=32)  # the body
    m.cushion("bottle_blue", (0.0, 0.0, 1.34), (0.135, 0.135, 0.035), e=0.9, nu=32, nv=8)  # its base, now on top
    for z in (1.14, 1.27):
        m.cylinder("bottle_blue", (0.0, 0.0, z), 0.1375, 0.007, seg=32, caps=False)  # the ribs
    m.cylinder("cooler_water", (0.0, 0.0, 1.02), 0.026, 0.03, seg=20, radius_top=0.125)  # the water
    m.cylinder("cooler_water", (0.0, 0.0, 1.16), 0.13, 0.11, seg=32)
    return m.write("proc_water_cooler", "Water cooler (procedural)",
                   "members: the cabinet box and the bottle cylinder")


def main() -> int:
    pf.OUT.mkdir(parents=True, exist_ok=True)
    for fn in (storage_pallet, storage_carton, storage_carton_top, wall_clock, desk_set, whiteboard, socket,
               light_switch, thermostat, exit_sign, first_aid_kit, notice_board, safety_poster, trunking,
               smoke_detector, print_corridor_a, print_corridor_b, print_reception, television, extinguisher,
               fire_sign, call_point, dome_camera, lab_bench_set, lab_island_set, shelf_loads, water_cooler):
        rec = fn()
        faces = sum(p["faces"] for p in rec["parts"])
        print(f"{rec['id']:22s} {faces:5d} faces  size {rec['size_m']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
