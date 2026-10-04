"""Paint the cats' coats on the fur texture of the converted cat (tools/convert_cat.py).
Development-only (needs requirements-dev.txt: scipy, pillow).

    .venv\\Scripts\\python tools\\make_coats.py

Each coat is defined on the cat's 3D body (bind pose), not on the texture, so its stripes and
patches run continuously across texture seams and follow the body: every texel of the fur map
gets its 3D position and body region from the mesh it covers. From the original fur only the fine,
neutral hair detail is kept (low-frequency shading and the original stripes are removed), and the
nose and paw pads keep their original colour. The source model is "Cat" by Vr-cvantorium (CC BY
4.0); these coats are modified versions of its texture and stay under CC BY 4.0.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from robot_env.cat_rig import CAT_DIR, load_rig  # noqa: E402

FUR_MATERIAL = "anisotropic1"
SIZE = 1024
NOSE_DEPTH = 0.006  # m: the front of the head that is nose leather
NOSE_COLOURS = {"ginger_mackerel": (0.80, 0.52, 0.50), "black_tuxedo": (0.20, 0.17, 0.18), "grey_tabby": (0.62, 0.48, 0.49)}
COATS = ("brown_tabby", "ginger_mackerel", "black_tuxedo", "grey_tabby")


def texel_geometry(rig, size: int = SIZE):
    """Per texel of the fur map: 3D bind position, per-joint weights, and coverage (inside a
    triangle). Barycentric rasterization of every fur triangle in texture space."""
    prim = rig.prim_material.index(FUR_MATERIAL)
    faces = rig.faces[rig.face_prim == prim]
    uv = rig.uv * [size, size]  # texel coordinates (v down, as in the image)
    nj = len(rig.joint_names)
    wfull = np.zeros((len(rig.vertices), nj))
    np.add.at(wfull, (np.repeat(np.arange(len(rig.vertices)), 4), rig.joints.reshape(-1)), rig.weights.reshape(-1))
    pos = np.zeros((size, size, 3))
    wts = np.zeros((size, size, nj), np.float32)
    cover = np.zeros((size, size), bool)
    for tri in faces:
        p = uv[tri]
        x0, y0 = np.floor(p.min(0)).astype(int)
        x1, y1 = np.ceil(p.max(0)).astype(int)
        x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, size - 1), min(y1, size - 1)
        if x1 < x0 or y1 < y0:
            continue
        xs, ys = np.meshgrid(np.arange(x0, x1 + 1) + 0.5, np.arange(y0, y1 + 1) + 0.5)
        d = (p[1, 1] - p[2, 1]) * (p[0, 0] - p[2, 0]) + (p[2, 0] - p[1, 0]) * (p[0, 1] - p[2, 1])
        if abs(d) < 1e-12:
            continue
        a = ((p[1, 1] - p[2, 1]) * (xs - p[2, 0]) + (p[2, 0] - p[1, 0]) * (ys - p[2, 1])) / d
        b = ((p[2, 1] - p[0, 1]) * (xs - p[2, 0]) + (p[0, 0] - p[2, 0]) * (ys - p[2, 1])) / d
        g = 1 - a - b
        inside = (a >= -0.02) & (b >= -0.02) & (g >= -0.02)
        if not inside.any():
            continue
        iy, ix = (ys[inside] - 0.5).astype(int), (xs[inside] - 0.5).astype(int)
        bar = np.stack([a[inside], b[inside], g[inside]], 1)
        pos[iy, ix] = bar @ rig.vertices[tri]
        wts[iy, ix] = bar @ wfull[tri]
        cover[iy, ix] = True
    return pos, wts, cover


def regions(rig, wts):
    """Body-region weights per texel from the skin weights."""
    def w(*names):
        idx = [i for i, n in enumerate(rig.joint_names) if any(k in n for k in names)]
        return wts[..., idx].sum(-1)
    return {
        "head": w("Head", "Eye", "Neck1"), "neck": w("Neck_022"),
        "tail": w("Tail"), "paw": w("Hand", "ToeBase", "Foot"),
        "leg": w("UpLeg", "Leg_0", "Arm", "ForeArm", "Shoulder"),
        "spine": w("Spine", "Hips"),
    }


def fur_detail(img: np.ndarray, cover: np.ndarray) -> np.ndarray:
    """Neutral, fine hair detail from the original fur: luminance minus its local average
    (removes shading and the painted stripes), limited to a gentle amplitude."""
    lum = img.astype(float) @ [0.299, 0.587, 0.114] / 255.0
    fine = lum - ndimage.gaussian_filter(lum, 2.0)
    fine /= max(np.percentile(np.abs(fine[cover]), 95), 1e-6)
    return np.clip(fine, -1.5, 1.5) * cover


def _noise(pos, scale, seed):
    rng = np.random.default_rng(seed)
    grid = rng.normal(size=(24, 24, 24))
    grid = ndimage.gaussian_filter(grid, 1.5)
    idx = np.clip((pos - pos.min((0, 1)) ) / scale, 0, 23)
    return ndimage.map_coordinates(grid, idx.reshape(-1, 3).T, order=1).reshape(pos.shape[:2])


def coat_colour(name: str, pos, reg, cover):
    """Base colour per texel for a coat (before fur detail), from 3D patterns and masks."""
    x, y, z = pos[..., 0], pos[..., 1], pos[..., 2]
    limb = np.clip((reg["leg"] + reg["paw"] - 0.3) / 0.5, 0, 1)  # 0 on the body, 1 on the legs (smooth)
    tail = reg["tail"]
    belly = np.clip((0.15 - z) / 0.05, 0, 1) * np.clip(1 - reg["leg"] - reg["paw"], 0, 1) * (1 - tail)
    chest = np.clip((x - 0.05) / 0.06, 0, 1) * np.clip((0.21 - z) / 0.05, 0, 1) * (1 - reg["head"])
    muzzle = reg["head"] * np.clip((x - 0.19) / 0.03, 0, 1) * np.clip((0.27 - z) / 0.03, 0, 1)
    paws = np.clip(reg["paw"] * 1.6, 0, 1)
    n = _noise(pos, 0.03, {"brown_tabby": 1, "ginger_mackerel": 2, "black_tuxedo": 3, "grey_tabby": 4}[name])

    n2 = _noise(pos, 0.012, 7)  # finer wobble

    def stripes(freq, width, phase=0.0, wobble=0.6):
        """Mackerel stripes: thin wavy rings around the body (along x), the legs (along z), and the
        tail, blended smoothly between body and legs; softer on the face, gone on the belly."""
        u = x * (1 - limb) + (z * 1.5 + 0.3) * limb
        u = u * (1 - tail) + x * 1.25 * tail
        s = np.sin(2 * np.pi * (u * freq + phase) + wobble * (2.5 * n + 1.2 * n2))
        # irregular width: broad along the back, thin toward the belly, varying with noise
        wid = width * (0.55 + 0.9 * np.clip((z - 0.12) / 0.12, 0, 1)) * (0.75 + 0.5 * np.clip(n2 + 0.5, 0, 1))
        band = np.clip((s - (1 - 2 * wid)) / (2 * wid + 1e-6), 0, 1) ** 0.7
        band *= np.clip(1.4 - 2.2 * np.abs(_noise(pos, 0.02, 11)), 0, 1)  # broken, interrupted stripes
        dorsal = np.exp(-(y / 0.012) ** 2) * np.clip((z - 0.20) / 0.03, 0, 1) * (1 - reg["head"]) * (1 - limb)
        return np.clip(band * (1 - 0.6 * reg["head"]) + dorsal, 0, 1)

    def mix(a, b, t):
        return a + (b - a) * t[..., None]

    if name == "ginger_mackerel":
        base = mix(np.array([0.80, 0.55, 0.33]), np.array([0.88, 0.67, 0.44]), np.clip(n + 0.5, 0, 1))
        col = mix(base, np.array([0.66, 0.40, 0.21]), 0.8 * stripes(30, 0.20) * (1 - belly))
        col = mix(col, np.array([0.95, 0.84, 0.66]), np.clip(belly + chest + muzzle * 0.8, 0, 1))
    elif name == "black_tuxedo":
        # not flat black: a real black coat has a warm brownish sheen that varies over the body
        col = mix(np.array([0.040, 0.038, 0.040]), np.array([0.105, 0.085, 0.072]), np.clip(n + 0.5, 0, 1) ** 1.5)
        # soft, slightly asymmetric white: the edges wander with noise and lean to one side
        edge = 0.6 * _noise(pos, 0.015, 21)
        socks = np.clip((0.065 + 0.012 * np.sign(y) + 0.01 * edge - z) / 0.02, 0, 1) * np.clip(reg["paw"] + reg["leg"], 0, 1)
        chest2 = np.clip((x - 0.05 + 0.02 * edge - 0.015 * y / 0.05) / 0.07, 0, 1) * np.clip((0.21 - z + 0.01 * edge) / 0.06, 0, 1)             * (1 - reg["head"])
        muzzle2 = reg["head"] * np.clip((x - 0.195 + 0.008 * edge + 0.01 * y / 0.03) / 0.04, 0, 1)             * np.clip((0.272 - z) / 0.04, 0, 1)
        white = np.clip(belly * 1.1 + chest2 * 1.2 + muzzle2 * 1.1 + socks + paws * 0.5, 0, 1)
        white = ndimage.gaussian_filter(white, 1.5)
        col = mix(col, np.array([0.92, 0.91, 0.89]), white)
    elif name == "grey_tabby":
        base = mix(np.array([0.53, 0.52, 0.50]), np.array([0.63, 0.62, 0.60]), np.clip(n + 0.5, 0, 1))
        col = mix(base, np.array([0.33, 0.32, 0.31]), 0.75 * stripes(26, 0.18, 0.3, 0.9) * (1 - belly))
        col = mix(col, np.array([0.90, 0.90, 0.88]), np.clip(belly * 1.3, 0, 1))
    else:
        raise ValueError(name)
    return col


def pads_mask(img: np.ndarray, reg, cover) -> np.ndarray:
    """Original paw-pad pixels (pinkish skin) kept as they are."""
    f = img.astype(float) / 255.0
    r, g, b = f[..., 0], f[..., 1], f[..., 2]
    pink = (r > 0.45) & (r - g > 0.12) & (r - b > 0.08) & (g < 0.62)
    return ndimage.binary_opening(pink & (reg["paw"] > 0.4) & cover, iterations=1)


def nose_mask(reg, cover, pos) -> np.ndarray:
    """The nose leather: the front NOSE_DEPTH of the head (soft edge). The source's own nose and
    cheek pixels are brown tabby fur, so each coat paints its own nose colour here."""
    head = (reg["head"] > 0.6) & cover
    front = pos[..., 0][head].max()
    return np.clip((pos[..., 0] - (front - NOSE_DEPTH)) / (NOSE_DEPTH * 0.5), 0, 1) * head


def dilate_into_background(img: np.ndarray, cover: np.ndarray, rounds: int = 8) -> np.ndarray:
    """Copy island-edge colours outward so texture filtering never mixes in background."""
    out, filled = img.copy(), cover.copy()
    for _ in range(rounds):
        grown = ndimage.binary_dilation(filled) & ~filled
        if not grown.any():
            break
        acc = np.zeros_like(out, float)
        cnt = np.zeros(out.shape[:2])
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            src = np.roll(filled, (dy, dx), (0, 1))
            acc += np.roll(out, (dy, dx), (0, 1)) * src[..., None]
            cnt += src
        out[grown] = (acc[grown] / np.maximum(cnt[grown], 1)[:, None])
        filled |= grown
    return out


def main() -> int:
    rig = load_rig()
    src = np.array(Image.open(CAT_DIR / f"cat_{FUR_MATERIAL}.png").convert("RGB"))
    pos, wts, cover = texel_geometry(rig)
    reg = regions(rig, wts)
    detail = fur_detail(src, cover)
    keep = pads_mask(src, reg, cover)
    nose = nose_mask(reg, cover, pos)
    made = {"brown_tabby": f"cat_{FUR_MATERIAL}.png"}  # the source coat itself
    for name in COATS[1:]:
        col = coat_colour(name, pos, reg, cover)
        amp = 0.18 if name != "black_tuxedo" else 0.45  # fur detail must show on a dark coat too
        col = col + (np.array(NOSE_COLOURS[name]) - col) * nose[..., None]
        col = np.clip(col * (1 + amp * detail[..., None]), 0, 1)
        out = np.where(keep[..., None], src / 255.0, col)
        out = np.where(cover[..., None], out, 0.0)
        out = dilate_into_background(out, cover)
        fname = f"coat_{name}.png"
        Image.fromarray((out * 255).round().astype(np.uint8)).save(CAT_DIR / fname)
        made[name] = fname
    manifest = json.loads((CAT_DIR / "manifest.json").read_text(encoding="utf-8"))
    manifest["coats"] = {"command": "python tools/make_coats.py", "files": made,
                         "note": "coats painted from 3D body patterns; only fine hair detail kept from the source"}
    (CAT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    print(made)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
