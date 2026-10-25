"""The "perfect map" comparison: the simulator's own floor plan written as a saved map.

    python -m AD.models.mapplan.floorplan --out floorplan

A real robot never has this map (it is the simulator's exact obstacle layout, furniture
included); it exists only to measure how much map quality matters: run the same model once with
the robot-made map (build.py) and once with this one. Same grid and file format as the robot-made
map, so MapDriver and MapRouteSensor load either with map_path=.../runs/<out>/map.npz.

A cell is occupied when its centre lies within OCC_MARGIN of an obstacle in the floor plan (the
robot's planner shapes, robot_env.layout), free when it is clear by more than that inside the
outer walls, and never seen outside them."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .savedmap import CELL, HALF, N, SavedMap

RUNS = Path(__file__).resolve().parent / "runs"
OCC_MARGIN = 0.5 * CELL * 2 ** 0.5  # half a cell diagonal: any obstacle touching the cell


def floor_plan_map() -> SavedMap:
    from robot_env import config as C
    from robot_env.layout import RoomMap, clearance
    from robot_env.system import RobotSystem

    system = RobotSystem(cats=0)
    shapes = RoomMap(system.sim.model).shapes
    system.close()
    c = (np.arange(N) + 0.5) * CELL - HALF
    x, y = np.meshgrid(c, c, indexing="ij")
    clear = clearance(shapes, x, y)
    inside = (np.abs(x) < C.FLOOR_HALF_SIZE) & (np.abs(y) < C.FLOOR_HALF_SIZE)
    m = SavedMap.empty()
    m.occupied = clear <= OCC_MARGIN
    m.free = inside & ~m.occupied
    m.occ_votes = m.occupied.astype(np.int32) * 100  # as if seen by 100 drives: finish() keeps it
    m.free_votes = m.free.astype(np.int32) * 100
    return m


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", default="floorplan")
    a = p.parse_args(argv)
    m = floor_plan_map()
    out = RUNS / a.out
    m.save(out / "map.npz")
    info = {"source": "simulator floor plan (comparison only, not a realistic map)", "occupied_cells": int(m.occupied.sum()),
            "free_cells": int(m.free.sum()), "cells": N * N, "cell_m": CELL}
    (out / "report.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
