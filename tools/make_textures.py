"""Generate the procedural textures used by robot_env/world.xml (no downloads).

    .venv\\Scripts\\python tools\\make_textures.py

Writes small PNG files to robot_env/assets/. Output is deterministic (fixed seeds).
"""

from pathlib import Path

import numpy as np
import pygame

OUT = Path(__file__).resolve().parent.parent / "robot_env" / "assets"


def save(name: str, rgb: np.ndarray) -> None:
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    surf = pygame.surfarray.make_surface(rgb.swapaxes(0, 1))
    pygame.image.save(surf, str(OUT / name))


def smooth_noise(rng, h, w, scale):
    """Value noise: random grid upsampled with bilinear interpolation."""
    gh, gw = max(2, h // scale + 2), max(2, w // scale + 2)
    grid = rng.random((gh, gw))
    y = np.linspace(0, gh - 1.001, h)
    x = np.linspace(0, gw - 1.001, w)
    y0, x0 = y.astype(int), x.astype(int)
    fy, fx = (y - y0)[:, None], (x - x0)[None, :]
    a = grid[y0][:, x0]
    b = grid[y0][:, x0 + 1]
    c = grid[y0 + 1][:, x0]
    d = grid[y0 + 1][:, x0 + 1]
    return a * (1 - fx) * (1 - fy) + b * fx * (1 - fy) + c * (1 - fx) * fy + d * fx * fy


def wood_floor(rng, size=1024, planks=8):
    """Oak-like planks: 8 rows per texture (about 0.15 m each at the world's scale),
    staggered end joints, fine grain stretched along the plank, slight tone variation."""
    h = w = size
    img = np.zeros((h, w, 3))
    plank_h = h // planks
    base = np.array([168, 128, 88])
    for p in range(planks):
        y0 = p * plank_h
        rows = slice(y0, y0 + plank_h)
        fine = smooth_noise(rng, plank_h, w // 16, 2)
        fine = np.repeat(fine, 16, axis=1)[:, :w]  # grain runs along the plank
        broad = smooth_noise(rng, plank_h, w, 64)
        joints = sorted(int(j) for j in rng.integers(w // 8, w - w // 8, size=2))
        edges = [0] + joints + [w]
        for a, b in zip(edges[:-1], edges[1:]):  # each board segment has its own tone
            tint = base * rng.uniform(0.86, 1.08)
            img[rows, a:b] = tint * (0.86 + 0.12 * fine[:, a:b, None] + 0.08 * broad[:, a:b, None])
            img[rows, a:a + 2] *= 0.72  # end joint
        img[y0:y0 + 2] *= 0.7  # long seam
    return img


def tiles(rng, size=512, n=4, color=(206, 204, 198), grout=(178, 176, 170)):
    """Large-format tiles with thin, light grout and slight per-tile variation."""
    img = np.ones((size, size, 3)) * np.array(color)
    step = size // n
    for i in range(n):
        for j in range(n):
            img[i * step:(i + 1) * step, j * step:(j + 1) * step] *= rng.uniform(0.97, 1.02)
    img *= (0.97 + 0.04 * smooth_noise(rng, size, size, 48))[..., None]
    for i in range(n + 1):
        img[max(0, i * step - 1):i * step + 1, :] = grout
        img[:, max(0, i * step - 1):i * step + 1] = grout
    return img


def carpet(rng, size=256):
    n = rng.random((size, size)) * 0.5 + smooth_noise(rng, size, size, 8) * 0.5
    return np.array([88, 98, 112]) * (0.8 + 0.35 * n[..., None])


def plaster(rng, size=256):
    """Painted plaster: almost uniform, with very subtle low-frequency variation."""
    n = smooth_noise(rng, size, size, 96) * 0.7 + smooth_noise(rng, size, size, 24) * 0.3
    return np.array([234, 231, 224]) * (0.975 + 0.025 * n[..., None])


def door_wood(rng, size=256):
    grain = smooth_noise(rng, size, size // 8, 4)
    grain = np.repeat(grain, 8, axis=1)[:, :size]
    img = np.array([150, 104, 66]) * (0.85 + 0.25 * grain[..., None])
    img[:, :6] *= 0.8
    img[:, -6:] *= 0.8
    return img


def label(text: str, w=512, h=128) -> np.ndarray:
    pygame.font.init()
    font = pygame.font.SysFont("segoeui,arial", 78, bold=True)
    surf = pygame.Surface((w, h))
    surf.fill((44, 62, 80))
    pygame.draw.rect(surf, (220, 226, 232), surf.get_rect(), 6)
    t = font.render(text, True, (240, 244, 248))
    surf.blit(t, ((w - t.get_width()) // 2, (h - t.get_height()) // 2))
    return pygame.surfarray.array3d(surf).swapaxes(0, 1).astype(float)


def outdoor_view(rng, w=512, h=512):
    """Window view: sky gradient, soft clouds, distant tree line, lawn."""
    y = np.linspace(0, 1, h)[:, None]
    sky = np.array([150, 190, 230]) * (1 - y[..., None] * 0.35) + np.array([60, 60, 40]) * y[..., None] * 0.2
    img = np.broadcast_to(sky, (h, w, 3)).copy()
    clouds = 0.6 * smooth_noise(rng, h, w, 128) + 0.4 * smooth_noise(rng, h, w, 40)
    fade = np.clip(1 - y / 0.6, 0, 1)  # clouds thin out toward the horizon
    img += (np.clip(clouds - 0.45, 0, 1) * 260 * fade)[..., None]
    trees = (0.62 + 0.06 * smooth_noise(rng, 1, w, 20)[0])
    rows = np.arange(h)[:, None] / h
    tree_mask = rows > trees[None, :]
    img[tree_mask] = (np.array([46, 82, 50]) * (0.8 + 0.3 * smooth_noise(rng, h, w, 6)[..., None]))[tree_mask]
    img[rows[:, 0] > 0.8] = np.array([92, 140, 70]) * (0.85 + 0.2 * smooth_noise(rng, h, w, 10)[rows[:, 0] > 0.8][..., None])
    return img


def rug(rng, w=512, h=320, base=(120, 40, 44), accent=(214, 180, 120)):
    img = np.ones((h, w, 3)) * np.array(base) * (0.9 + 0.15 * smooth_noise(rng, h, w, 3)[..., None])
    for m in (14, 30):
        img[m:m + 6, m:w - m] = accent
        img[h - m - 6:h - m, m:w - m] = accent
        img[m:h - m, m:m + 6] = accent
        img[m:h - m, w - m - 6:w - m] = accent
    cy, cx = h // 2, w // 2
    yy, xx = np.mgrid[0:h, 0:w]
    diamond = (np.abs(yy - cy) / (h * 0.28) + np.abs(xx - cx) / (w * 0.28)) < 1
    img[diamond] = img[diamond] * 0.6 + np.array(accent) * 0.4
    return img


def painting(rng, w=384, h=256):
    img = np.ones((h, w, 3)) * np.array([238, 232, 220])
    palette = [np.array(c) for c in ((196, 82, 60), (60, 104, 150), (230, 180, 70), (70, 120, 90), (40, 40, 48))]
    for _ in range(9):
        x0, y0 = rng.integers(0, w - 60), rng.integers(0, h - 40)
        x1, y1 = x0 + rng.integers(40, w // 2), y0 + rng.integers(30, h // 2)
        img[y0:y1, x0:x1] = palette[rng.integers(len(palette))] * rng.uniform(0.85, 1.05)
    img[:10], img[-10:], img[:, :10], img[:, -10:] = 30, 30, 30, 30
    return img


def whiteboard(rng, w=512, h=256):
    img = np.ones((h, w, 3)) * 246
    for k in range(7):  # marker strokes
        y = 30 + k * 28
        x = 30
        color = np.array((40, 60, 150) if k % 3 else (180, 40, 40))
        while x < w - 60 - rng.integers(0, 200):
            seg = int(rng.integers(12, 40))
            img[y:y + 3, x:x + seg] = color
            x += seg + int(rng.integers(6, 14))
    img[:6], img[-6:], img[:, :6], img[:, -6:] = 170, 170, 170, 170
    return img


def tv_screen(rng, w=512, h=288):
    y = np.linspace(0, 1, h)[:, None, None]
    img = np.ones((h, w, 3)) * np.array([14, 16, 20]) + y * np.array([10, 12, 16])
    img[: h // 3, : w // 2] += 18  # soft reflection
    return img


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    save("floor_wood.png", wood_floor(rng))
    save("floor_tile.png", tiles(rng))
    save("floor_tile_dark.png", tiles(rng, color=(128, 130, 134), grout=(108, 110, 114)))
    save("carpet.png", carpet(rng))
    save("plaster.png", plaster(rng))
    save("door_wood.png", door_wood(rng))
    save("outdoor_view.png", outdoor_view(rng))
    save("rug_red.png", rug(rng))
    save("rug_blue.png", rug(rng, base=(46, 70, 108), accent=(220, 210, 190)))
    save("painting_1.png", painting(rng))
    save("painting_2.png", painting(rng))
    save("whiteboard.png", whiteboard(rng))
    save("tv_screen.png", tv_screen(rng))
    for name in ("OFFICE", "LAB", "STORAGE", "RECEPTION"):
        save(f"label_{name.lower()}.png", label(name))
    print("textures written to", OUT)


if __name__ == "__main__":
    main()
