"""Fit stacked cylinders to a round part of a furniture model (a pot, a bin): prints the centre
and the bands to paste into tools/build_world.py (development tool).

    .venv\\Scripts\\python tools\\fit_round.py potted_plant_01 0.535 [max_taper]

The centre is the middle of the part's extent; each band's radius is the largest radius seen in
it (so the mesh never sticks out), and a band ends when the radius within it varies by more than
max_taper (default 1.5 cm).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import furniture_geom as fg  # noqa: E402

STEP = 0.0025


def fit(model_id: str, top: float, max_taper: float = 0.015):
    v, f = fg.model_mesh(model_id)
    lo, hi = v[:, :2].min(0) - 0.01, v[:, :2].max(0) + 0.01
    zs = np.arange(STEP / 2, top, STEP)
    cells = []
    for z in zs:
        g = fg.slice_raster(v, f, z, lo, hi)
        cells.append(lo + (np.argwhere(g) + 0.5) * 0.005)
    every = np.vstack([c for c in cells if len(c)])
    centre = (every.min(0) + every.max(0)) / 2
    rs, rmin = [], []  # per height: the largest radius, and the smallest of the 18 sectors' largest
    for c in cells:
        if not len(c):
            rs.append(0.0)
            rmin.append(0.0)
            continue
        d = np.hypot(*(c - centre).T)
        a = np.degrees(np.arctan2(c[:, 1] - centre[1], c[:, 0] - centre[0]))
        sect = [d[(a >= k) & (a < k + 20)].max() for k in range(-180, 180, 20) if np.any((a >= k) & (a < k + 20))]
        rs.append(float(d.max()) + 0.0025)
        rmin.append(float(min(sect)) - 0.0025)
    rs, rmin = np.array(rs), np.array(rmin)
    bands, i = [], 0
    while i < len(zs):  # the member's radius minus the mesh's nearest surface stays within max_taper
        j, lo_r, hi_r = i, rmin[i], rs[i]
        while j + 1 < len(zs) and max(hi_r, rs[j + 1]) - min(lo_r, rmin[j + 1]) <= max_taper:
            j += 1
            lo_r, hi_r = min(lo_r, rmin[j]), max(hi_r, rs[j])
        z0 = 0.0 if i == 0 else round(float(zs[i] - STEP / 2), 4)
        z1 = round(float(min(top, zs[j] + STEP / 2)), 4)
        bands.append((z0, z1, round(float(hi_r), 4)))
        i = j + 1
    return tuple(round(float(c), 4) for c in centre), tuple(bands)


if __name__ == "__main__":
    mid, top = sys.argv[1], float(sys.argv[2])
    taper = float(sys.argv[3]) if len(sys.argv) > 3 else 0.015
    c, b = fit(mid, top, taper)
    print("centre", c)
    print("bands", b)
