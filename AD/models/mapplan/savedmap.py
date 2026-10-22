"""The saved building map: a 0.1 m grid in the building's own frame (the frame task start poses
are given in), with three states per cell: occupied, free, or never seen.

It is made by the robot itself (build.py): the built-in driver drives mapping tasks with its own
lidar, depth and odometry, and each drive's map is placed in the building frame with that drive's
start pose (the pose an operator gives a real robot when it is switched on). Nothing is read from
the simulator's floor plan to make it; build.py only compares against the floor plan afterwards,
to report how accurate the map is."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from robot_env.baseline import CELL, SIZE

HALF = 6.0  # m: the map covers -HALF..HALF on both axes (the floor is 10 x 10 m)
N = int(round(2 * HALF / CELL))  # cells per side
DEFAULT = Path(__file__).resolve().parent / "runs" / "map_v1" / "map.npz"


@dataclass
class SavedMap:
    occupied: np.ndarray  # (N, N) bool
    free: np.ndarray  # (N, N) bool: seen free and not occupied
    occ_votes: np.ndarray  # (N, N) int: drives that saw the cell occupied
    free_votes: np.ndarray  # (N, N) int: drives that saw it free

    @classmethod
    def empty(cls) -> "SavedMap":
        z = np.zeros((N, N), np.int32)
        return cls(np.zeros((N, N), bool), np.zeros((N, N), bool), z.copy(), z.copy())

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, occupied=self.occupied, free=self.free, occ_votes=self.occ_votes,
                            free_votes=self.free_votes, cell=CELL, half=HALF)

    @classmethod
    def load(cls, path: Path | str | None = None) -> "SavedMap":
        f = Path(path) if path else DEFAULT
        if not f.exists():
            raise FileNotFoundError(f"no saved map at {f}: build it first (python -m AD.models.mapplan.build)")
        d = np.load(f)
        if float(d["cell"]) != CELL or float(d["half"]) != HALF:
            raise ValueError(f"map grid {float(d['cell'])} m / {float(d['half'])} m does not match {CELL} / {HALF}")
        return cls(d["occupied"], d["free"], d["occ_votes"], d["free_votes"])

    def finish(self, min_votes: int = 2, min_share: float = 0.3) -> None:
        """Turn the votes into the map: occupied when seen occupied by at least min_votes drives
        and in at least min_share of the drives that saw the cell at all; free when seen free."""
        seen = self.occ_votes + self.free_votes
        self.occupied = (self.occ_votes >= min_votes) & (self.occ_votes >= min_share * np.maximum(seen, 1))
        self.free = (self.free_votes > 0) & ~self.occupied


def world_index(wx: np.ndarray, wy: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Building-frame points to map cells, with an in-bounds mask."""
    i = np.floor((np.asarray(wx) + HALF) / CELL).astype(int)
    j = np.floor((np.asarray(wy) + HALF) / CELL).astype(int)
    return i, j, (i >= 0) & (i < N) & (j >= 0) & (j < N)


def driver_cells_in_world(pose0: tuple[float, float, float]) -> tuple[np.ndarray, np.ndarray]:
    """Building-frame centres of every cell of a driver's grid, for a drive that started at pose0
    (the driver's grid is centred on its start, x along the start heading)."""
    x0, y0, yaw = pose0
    c = (np.arange(SIZE) - SIZE // 2 + 0.5) * CELL
    sx, sy = np.meshgrid(c, c, indexing="ij")
    cs, sn = math.cos(yaw), math.sin(yaw)
    return x0 + cs * sx - sn * sy, y0 + sn * sx + cs * sy


def to_driver_frame(m: SavedMap, pose0: tuple[float, float, float]) -> tuple[np.ndarray, np.ndarray]:
    """The saved map resampled onto a driver's grid for a drive starting at pose0: (occupied,
    free), each (SIZE, SIZE). Each driver cell takes the map cell its centre falls in."""
    wx, wy = driver_cells_in_world(pose0)
    i, j, ok = world_index(wx, wy)
    occ = np.zeros((SIZE, SIZE), bool)
    free = np.zeros((SIZE, SIZE), bool)
    occ[ok] = m.occupied[i[ok], j[ok]]
    free[ok] = m.free[i[ok], j[ok]]
    return occ, free
