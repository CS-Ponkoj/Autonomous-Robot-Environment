"""Download CC0 surface textures (Poly Haven, ambientCG) and prepare them for MuJoCo's classic renderer.
Development tool: the prepared files are committed under robot_env/assets/textures/.

    .venv\\Scripts\\python tools\\fetch_textures.py            (every texture in TEXTURES)
    .venv\\Scripts\\python tools\\fetch_textures.py --check    (prepare again: byte-identical?)

The classic renderer uses a material's base colour only (no normal, roughness, or occlusion
maps), so each texture's relief is baked into its colour: the diffuse map is multiplied by its
ambient occlusion and by a soft shading from its normal map, as if lit from above and slightly
in front (RELIEF), then resized to the size given here. The source files are downloaded once
into the texture cache (never published) and checked against Poly Haven's MD5 sums; their
SHA-256 sums, the real-world size of one texture repeat (from Poly Haven), the licence, and
the tool's own hash are recorded in <id>.json, and MANIFEST.json lists every committed file.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import sys
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "robot_env" / "assets" / "textures"
API = "https://api.polyhaven.com"
AGENT = {"User-Agent": "autonomous-robot-environment-dev/1.0"}
AO_STRENGTH = 0.9  # 0: no occlusion, 1: the full map
RELIEF = 0.35  # how strongly the normal map's shading is baked in (0: none)
LIGHT = np.array([0.0, 0.35, 1.0]) / np.linalg.norm([0.0, 0.35, 1.0])  # from above, a little in front

# texture id -> how to prepare it. Only textures used in the world. source: "polyhaven" or
# "ambientcg"; size: output pixels (square); repeat_m: metres of floor or wall one repeat covers
# (from the source where it gives one, else measured off the image's own pattern); neutral:
# (mean, contrast) to turn the colour to grey of that mean, its variation scaled by contrast
# (the material's colour then tints it), or None to keep the source colour.
TEXTURES: dict[str, dict] = {
    # walls: a fine indoor roller-paint plaster, grey, its variation set (std 0.04 of full scale)
    "plastered_wall_04": dict(source="polyhaven", size=1024, neutral=(0.93, None, 0.04)),
    # window blinds: a woven hessian, light
    "hessian_230": dict(source="polyhaven", size=512, saturation=0.5, target_mean=0.8),
    "laminate_floor_02": dict(source="polyhaven", size=1024, neutral=None),
    "white_oak_veneer": dict(source="polyhaven", size=1024, neutral=None),
    "OfficeCeiling001": dict(source="ambientcg", size=1024, repeat_m=1.2, neutral=None),
    # carpet: laid as 0.5 m carpet tiles, quarter-turned, brighter and calmer than the source
    "Carpet012": dict(source="ambientcg", size=1024, repeat_m=1.0, neutral=None, gain=4.6, variation=0.6,
                      quarter_tiles=True, target_mean=0.57, saturation=0.55, tile_tones=(1.04, 0.97, 1.0, 0.96)),
    # porcelain: a 4 x 4 block of 0.5 m tiles from the 2K source (2 mm per texel), 2 m repeat
    "Tiles040": dict(source="ambientcg", res="2K", size=1024, repeat_m=2.0, crop=0.5, neutral=None, variation=0.7),
    "Concrete031": dict(source="ambientcg", res="2K", size=1024, repeat_m=4.0, neutral=None),
}


def cache_dir() -> Path:
    base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~/.cache")
    d = Path(base) / "autonomous-robot-environment" / "texture_cache"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _get(url: str) -> bytes:
    with urllib.request.urlopen(urllib.request.Request(url, headers=AGENT), timeout=120) as r:
        return r.read()


def _json(url: str) -> dict:
    return json.loads(_get(url))


def download_ambientcg(tex_id: str, res: str = "1K") -> tuple[dict, dict]:
    """The colour, ambient occlusion, and OpenGL normal maps from ambientCG's JPG zip (res: 1K or
    2K), cached (the zip's SHA-256 is recorded; ambientCG publishes no checksum)."""
    import zipfile
    url = f"https://ambientcg.com/get?file={tex_id}_{res}-JPG.zip"
    target = cache_dir() / tex_id / f"{tex_id}_{res}-JPG.zip"
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(_get(url))
    zsha = hashlib.sha256(target.read_bytes()).hexdigest()
    out = {}
    with zipfile.ZipFile(target) as z:
        for key, suffix in (("Diffuse", "_Color.jpg"), ("AO", "_AmbientOcclusion.jpg"), ("nor_gl", "_NormalGL.jpg")):
            names = [n for n in z.namelist() if n.endswith(suffix)]
            if names:
                path = target.parent / names[0]
                path.write_bytes(z.read(names[0]))
                out[key] = (path, hashlib.sha256(path.read_bytes()).hexdigest(), f"{url}#{names[0]}")
    return out, {"authors": {"ambientCG": 1}, "zip_sha256": zsha}


def download(tex_id: str, res: str = "2k") -> dict:
    """The diffuse, ambient occlusion, and OpenGL normal maps (JPG), cached and MD5-checked.
    Returns {map: (path, sha256, url)} and the asset info."""
    files = _json(f"{API}/files/{tex_id}")
    info = _json(f"{API}/info/{tex_id}")
    out = {}
    for key in ("Diffuse", "AO", "nor_gl"):
        if key not in files:
            continue
        entry = files[key][res]["jpg"]
        target = cache_dir() / tex_id / Path(entry["url"]).name
        if not target.exists() or hashlib.md5(target.read_bytes()).hexdigest() != entry["md5"]:
            target.parent.mkdir(parents=True, exist_ok=True)
            data = _get(entry["url"])
            if hashlib.md5(data).hexdigest() != entry["md5"]:
                raise RuntimeError(f"{tex_id} {key}: MD5 mismatch")
            target.write_bytes(data)
        out[key] = (target, hashlib.sha256(target.read_bytes()).hexdigest(), entry["url"])
    return out, info


def prepare(maps: dict, size: int, neutral=None, gain: float = 1.0, crop: float = 1.0,
            variation: float = 1.0, quarter_tiles: bool = False, target_mean: float | None = None,
            saturation: float = 1.0, tile_tones=(1.0, 1.0, 1.0, 1.0)) -> Image.Image:
    """Diffuse x occlusion x normal-map shading, resized (LANCZOS) to size x size, sRGB; with
    `neutral` (mean, contrast) turned grey first, `crop` (the top-left fraction of the source
    kept: more texels per metre), `variation` (its light variation scaled around the mean), and
    `quarter_tiles` (laid as four carpet tiles, each turned a quarter from its neighbour, the
    seams between them a little darker), as TEXTURES gives them."""
    def load(path, mode):
        im = Image.open(path).convert(mode)
        if crop < 1.0:
            im = im.crop((0, 0, int(im.width * crop), int(im.height * crop)))
        return np.asarray(im, dtype=np.float64) / 255.0
    diff = load(maps["Diffuse"][0], "RGB")
    if neutral is not None:  # (mean, contrast) or (mean, None, std): grey, its variation set
        mean, contrast, *std = neutral
        g = diff @ np.array([0.2126, 0.7152, 0.0722])
        if contrast is None:
            contrast = std[0] / max(float(g.std()), 1e-6)
        g = mean + (g - g.mean()) * contrast
        diff = np.repeat(np.clip(g, 0.0, 1.0)[..., None], 3, axis=2)
    if saturation != 1.0:
        g = (diff @ np.array([0.2126, 0.7152, 0.0722]))[..., None]
        diff = np.clip(g + saturation * (diff - g), 0.0, 1.0)
    lin = diff ** 2.2 * gain  # work in linear light (gain: brighter than the source)
    if "AO" in maps:
        ao = load(maps["AO"][0], "L")
        lin *= (1.0 - AO_STRENGTH) + AO_STRENGTH * ao[..., None]
    if "nor_gl" in maps:
        n = load(maps["nor_gl"][0], "RGB") * 2.0 - 1.0
        n /= np.maximum(np.linalg.norm(n, axis=2, keepdims=True), 1e-6)
        shade = np.clip(n @ LIGHT, 0.0, 1.0) / LIGHT[2]  # 1 on a flat patch
        lin *= (1.0 - RELIEF) + RELIEF * np.clip(shade, 0.0, 1.6)[..., None]
    if variation != 1.0:
        mean = lin.mean(axis=(0, 1), keepdims=True)
        lin = mean + (lin - mean) * variation
    rgb = np.clip(lin, 0.0, 1.0) ** (1 / 2.2)
    if target_mean is not None:  # brightness set directly: the mean of the image (sRGB, 0 to 1)
        rgb = np.clip(rgb * (target_mean / max(float(rgb.mean()), 1e-6)), 0.0, 1.0)
    img = Image.fromarray((rgb * 255.0 + 0.5).astype(np.uint8), "RGB")
    if not quarter_tiles:
        return img.resize((size, size), Image.LANCZOS)
    half = size // 2
    tile = np.asarray(img.resize((half, half), Image.LANCZOS), dtype=np.float64)
    seam = np.ones((half, half, 1))
    seam[:2], seam[-2:], seam[:, :2], seam[:, -2:] = 0.82, 0.82, 0.82, 0.82  # the joint between tiles
    out = np.zeros((size, size, 3))
    for i in range(2):
        for j in range(2):
            out[i * half:(i + 1) * half, j * half:(j + 1) * half] = np.rot90(tile, k=(i + 2 * j) % 4) * seam * tile_tones[i + 2 * j]
    return Image.fromarray(np.clip(out + 0.5, 0, 255).astype(np.uint8), "RGB")


def _png(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def build(out: Path, wanted: list[str]) -> list[dict]:
    out.mkdir(parents=True, exist_ok=True)
    tool = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    records = []
    for tex_id in wanted:
        spec = TEXTURES[tex_id]
        if spec["source"] == "ambientcg":
            maps, info = download_ambientcg(tex_id, spec.get("res", "1K"))
            repeat = [spec["repeat_m"]] * 2
            page = f"https://ambientcg.com/view?id={tex_id}"
        else:
            maps, info = download(tex_id)
            dims = info.get("dimensions") or [None, None]
            repeat = [d / 1000.0 if d else None for d in dims[:2]]
            page = f"https://polyhaven.com/a/{tex_id}"
        data = _png(prepare(maps, spec["size"], spec.get("neutral"), spec.get("gain", 1.0), spec.get("crop", 1.0),
                            spec.get("variation", 1.0), spec.get("quarter_tiles", False), spec.get("target_mean"),
                            spec.get("saturation", 1.0), spec.get("tile_tones", (1.0, 1.0, 1.0, 1.0))))
        (out / f"{tex_id}.png").write_bytes(data)
        rec = {"id": tex_id, "file": f"{tex_id}.png", "size_px": spec["size"], "repeat_m": repeat,
               "neutral": spec.get("neutral"), "licence": "CC0 1.0", "source": page,
               "authors": sorted((info.get("authors") or {}).keys()),
               "inputs": {k: {"url": u, "sha256": h} for k, (_, h, u) in maps.items()},
               "ao_strength": AO_STRENGTH, "relief": RELIEF, "tool_sha256": tool,
               "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
        (out / f"{tex_id}.json").write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8")
        records.append(rec)
        print(tex_id, len(data) // 1024, "KiB")
    manifest = {p.name: {"sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "bytes": p.stat().st_size}
                for p in sorted(out.glob("*")) if p.name != "MANIFEST.json"}
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    return records


# Rugs: the existing designs (robot_env/assets/rug_*.png, tools/make_textures.py) woven into a CC0
# fibre texture: name -> (design file, rug size in m, fibre texture, metres per fibre repeat).
RUGS = {"rug_red_woven": ("rug_red.png", (1.7, 1.2), "Carpet014", 0.4),
        "rug_blue_woven": ("rug_blue.png", (1.9, 1.3), "Carpet014", 0.4)}
RUG_FIBRE = 0.45  # how strongly the fibre's light and dark show (0: a flat print)
RUG_BINDING = 0.03  # m: the bound edge, a little darker


def make_rugs(out: Path) -> list[dict]:
    """Each rug's design multiplied by its fibre texture (normalized to mean 1, variation scaled by
    RUG_FIBRE, repeated at its real scale over the rug) with a darker bound edge; one image
    mapped once over the rug's top."""
    records = []
    for name, (design_file, (w, h), fibre_id, fibre_m) in RUGS.items():
        maps, info = download_ambientcg(fibre_id)
        fibre = np.asarray(Image.open(maps["Diffuse"][0]).convert("L"), dtype=np.float64) / 255.0
        fibre = 1.0 + (fibre / fibre.mean() - 1.0) * RUG_FIBRE
        px_w = 1024
        px_h = int(round(px_w * h / w))
        design = np.asarray(Image.open(ROOT / "robot_env" / "assets" / design_file).convert("RGB").resize((px_w, px_h), Image.LANCZOS),
                            dtype=np.float64) / 255.0
        reps = (max(1, int(round(w / fibre_m))), max(1, int(round(h / fibre_m))))
        tile = Image.fromarray((np.clip(fibre, 0, 2) * 127.5).astype(np.uint8)).resize(
            (max(1, px_w // reps[0]), max(1, px_h // reps[1])), Image.LANCZOS)
        tiled = np.tile(np.asarray(tile, dtype=np.float64) / 127.5, (reps[1] + 1, reps[0] + 1))[:px_h, :px_w]
        rug = design * tiled[..., None]
        edge = int(round(px_w * RUG_BINDING / w))
        rug[:edge] *= 0.78
        rug[-edge:] *= 0.78
        rug[:, :edge] *= 0.78
        rug[:, -edge:] *= 0.78
        data = _png(Image.fromarray((np.clip(rug, 0, 1) * 255 + 0.5).astype(np.uint8), "RGB"))
        (out / f"{name}.png").write_bytes(data)
        rec = {"id": name, "file": f"{name}.png", "design": design_file, "fibre": fibre_id,
               "fibre_source": f"https://ambientcg.com/view?id={fibre_id}", "licence": "CC0 1.0 (fibre); design: this project",
               "fibre_inputs": {k: {"url": u, "sha256": hsh} for k, (_, hsh, u) in maps.items()},
               "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
        (out / f"{name}.json").write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8")
        records.append(rec)
        print(name, len(data) // 1024, "KiB")
    return records


def main(argv: list[str]) -> int:
    if "--check" in argv:
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            build(Path(tmp), list(TEXTURES))
            same = all((Path(tmp) / p.name).read_bytes() == p.read_bytes() for p in OUT.glob("*.png"))
        print("byte-identical" if same else "DIFFERENT")
        return 0 if same else 1
    wanted = [a for a in argv if not a.startswith("--")]
    build(OUT, [a for a in wanted if a in TEXTURES] or ([] if wanted else list(TEXTURES)))
    if not wanted or any(a in RUGS for a in wanted):
        make_rugs(OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
