"""Download CC0 surface textures from Poly Haven and prepare them for MuJoCo's classic renderer.
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

# texture id -> output size in pixels (square). Only textures used in the world.
TEXTURES: dict[str, int] = {}


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


def prepare(maps: dict, size: int) -> Image.Image:
    """Diffuse x occlusion x normal-map shading, resized (LANCZOS) to size x size, sRGB."""
    diff = np.asarray(Image.open(maps["Diffuse"][0]).convert("RGB"), dtype=np.float64) / 255.0
    lin = diff ** 2.2  # work in linear light
    if "AO" in maps:
        ao = np.asarray(Image.open(maps["AO"][0]).convert("L"), dtype=np.float64) / 255.0
        lin *= (1.0 - AO_STRENGTH) + AO_STRENGTH * ao[..., None]
    if "nor_gl" in maps:
        n = np.asarray(Image.open(maps["nor_gl"][0]).convert("RGB"), dtype=np.float64) / 255.0 * 2.0 - 1.0
        n /= np.maximum(np.linalg.norm(n, axis=2, keepdims=True), 1e-6)
        shade = np.clip(n @ LIGHT, 0.0, 1.0) / LIGHT[2]  # 1 on a flat patch
        lin *= (1.0 - RELIEF) + RELIEF * np.clip(shade, 0.0, 1.6)[..., None]
    rgb = np.clip(lin, 0.0, 1.0) ** (1 / 2.2)
    img = Image.fromarray((rgb * 255.0 + 0.5).astype(np.uint8), "RGB")
    return img.resize((size, size), Image.LANCZOS)


def _png(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def build(out: Path, wanted: list[str]) -> list[dict]:
    out.mkdir(parents=True, exist_ok=True)
    tool = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    records = []
    for tex_id in wanted:
        maps, info = download(tex_id)
        data = _png(prepare(maps, TEXTURES[tex_id]))
        (out / f"{tex_id}.png").write_bytes(data)
        dims = info.get("dimensions") or [None, None]
        rec = {"id": tex_id, "file": f"{tex_id}.png", "size_px": TEXTURES[tex_id],
               "repeat_m": [d / 1000.0 if d else None for d in dims[:2]],
               "licence": "CC0 1.0", "source": f"https://polyhaven.com/a/{tex_id}",
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


def main(argv: list[str]) -> int:
    if "--check" in argv:
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            build(Path(tmp), list(TEXTURES))
            same = all((Path(tmp) / p.name).read_bytes() == p.read_bytes() for p in OUT.glob("*.png"))
        print("byte-identical" if same else "DIFFERENT")
        return 0 if same else 1
    build(OUT, [a for a in argv if not a.startswith("--")] or list(TEXTURES))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
