"""The view out of every window: perspective crops of one CC0 outdoor panorama (Poly Haven's
tonemapped JPG), each looking the way its window faces, so windows on one wall show neighbouring
parts of the same scene. Development tool: the crops are committed under robot_env/assets/textures/.

    .venv\\Scripts\\python tools\\fetch_view.py

Each crop is a pinhole view from 1.5 m inside the room through the window's opening (its field
of view from the window's size), centred on the horizon, rectilinear (straight lines stay
straight), resampled bilinearly, then brightened to EXPOSURE (the glass shows it emissive).
The panorama is downloaded once into the texture cache and checked against Poly Haven's MD5.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_textures import AGENT, API, OUT, _get, _json, _png, cache_dir  # noqa: E402

PANORAMA = "buikslotermeerplein"
VIEW_DISTANCE = 1.5  # m from the window a person stands (sets each window's field of view)
EXPOSURE = 1.3  # the crops are brightened by this: daylight outside outshines the room
WIDTH = 640  # px of each crop (height from the window's shape)
KNEE, ROLL = 0.66, 0.26  # highlight roll-off: linear up to KNEE, then easing toward KNEE + ROLL
# Panorama azimuth (degrees) seen looking north (+y); east (+x) is 90 degrees clockwise from it.
NORTH = 300.0


def windows() -> list[tuple[str, float, float, float, float]]:
    """(name, outward heading in degrees from north clockwise, offset along the wall in m,
    width, height) of every window, as tools/build_world.py places them."""
    import build_world as B
    return [(f"view_{k}", heading, along, w, h) for k, (heading, along, w, h) in enumerate(B.WINDOW_VIEWS)]


def crop(pano: np.ndarray, heading: float, along: float, w: float, h: float) -> np.ndarray:
    hfov = 2 * math.atan(w / 2 / VIEW_DISTANCE)
    vfov = 2 * math.atan(h / 2 / VIEW_DISTANCE)
    out_w = WIDTH
    out_h = max(2, int(round(WIDTH * h / w)))
    yaw = math.radians(heading + NORTH) + along / 12.0  # windows further along look a little further round
    xs = np.tan(np.linspace(-hfov / 2, hfov / 2, out_w))
    ys = np.tan(np.linspace(vfov / 2, -vfov / 2, out_h))
    X, Y = np.meshgrid(xs, ys)
    lon = yaw + np.arctan2(X, 1.0)
    lat = np.arctan2(Y, np.hypot(X, 1.0)) + math.radians(9.0)  # standing, a person sees more sky than ground
    ph, pw = pano.shape[:2]
    u = (lon / (2 * math.pi) % 1.0) * pw
    v = (0.5 - lat / math.pi) * ph
    channels = [map_coordinates(pano[..., c], [v, u], order=1, mode="wrap") for c in range(3)]
    img = np.stack(channels, axis=2) / 255.0
    img = img * EXPOSURE
    # highlights roll off smoothly to KNEE + ROLL (the room's light adds to the glass's own glow)
    over = np.maximum(img - KNEE, 0.0)
    return np.where(img > KNEE, KNEE + ROLL * (1.0 - np.exp(-over / ROLL)), img)


def main() -> int:
    entry = _json(f"{API}/files/{PANORAMA}")["tonemapped"]
    target = cache_dir() / PANORAMA / Path(entry["url"]).name
    if not target.exists() or hashlib.md5(target.read_bytes()).hexdigest() != entry["md5"]:
        target.parent.mkdir(parents=True, exist_ok=True)
        data = _get(entry["url"])
        if hashlib.md5(data).hexdigest() != entry["md5"]:
            raise RuntimeError("panorama MD5 mismatch")
        target.write_bytes(data)
    Image.MAX_IMAGE_PIXELS = None
    pano = np.asarray(Image.open(target).convert("RGB"), dtype=np.float32)
    tool = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    info = _json(f"{API}/info/{PANORAMA}")
    for name, heading, along, w, h in windows():
        img = Image.fromarray((crop(pano, heading, along, w, h) * 255 + 0.5).astype(np.uint8), "RGB")
        data = _png(img)
        (OUT / f"{name}.png").write_bytes(data)
        rec = {"id": name, "file": f"{name}.png", "panorama": PANORAMA, "heading_deg": heading, "along_m": along,
               "window_m": [w, h], "licence": "CC0 1.0", "source": f"https://polyhaven.com/a/{PANORAMA}",
               "authors": sorted((info.get("authors") or {}).keys()),
               "input": {"url": entry["url"], "sha256": hashlib.sha256(target.read_bytes()).hexdigest()},
               "tool_sha256": tool, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
        (OUT / f"{name}.json").write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8")
        print(name, img.size, len(data) // 1024, "KiB")
    manifest = {p.name: {"sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "bytes": p.stat().st_size}
                for p in sorted(OUT.glob("*")) if p.name != "MANIFEST.json"}
    (OUT / "MANIFEST.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
