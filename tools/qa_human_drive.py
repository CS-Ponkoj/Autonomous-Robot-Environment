"""QA: drive every held-out goal in the real window, in real time, with key events.

This is a PRIVILEGED SCRIPTED KEY-EVENT INTEGRATION TEST, not a human acceptance test.
Each frame a bot decides which of W, A, D, S to hold, using the true pose, the planned
path, and App.free_moves() (the same check the status panel shows). The keys are
posted as real pygame key events, so they pass through ManualInput, ManualDriver,
RobotSystem.apply, the safety layer, and the window loop exactly like a person's keys.
It does not look at rendered pixels and cannot judge usability; the owner's own drive
is still required.

    .venv\Scripts\python tools\qa_human_drive.py            # all held-out seeds
    .venv\Scripts\python tools\qa_human_drive.py 1003 1005  # chosen seeds
"""

import math
import sys
import time
from pathlib import Path

import pygame

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from robot_env import config as C  # noqa: E402
from robot_env.app import App  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "qa_output"


class KeyBot:
    def __init__(self) -> None:
        self.points = None
        self.blocked_since = None
        self.recover_until = 0.0
        self.recover_keys: set[int] = set()
        self.recoveries = 0

    def __call__(self, app: App) -> set[int]:
        if app.episode.status != "running":
            return set()
        if self.points is None:
            self.points = list(app.episode.task.path[1:])
        s = app.system
        if s.time < self.recover_until:
            return self.recover_keys
        res = s.last_result
        stuck = "clearance" in res.reasons and abs(res.command.v) < 0.02 and abs(res.command.omega) < 0.05
        if stuck:
            self.blocked_since = self.blocked_since if self.blocked_since is not None else s.time
            if s.time - self.blocked_since > 0.4:
                free = app.free_moves()  # what the status panel shows
                x, y, yaw = s.sim.true_pose()
                tx, ty = self.points[0]
                err = math.atan2(ty - y, tx - x) - yaw
                want = "A" if math.atan2(math.sin(err), math.cos(err)) > 0 else "D"
                if want in free:  # turn toward the target in place
                    key, secs = want, 0.6
                else:  # back up, then the next check can turn toward the target
                    key, secs = ("S", 0.8) if "S" in free else (("A" if "A" in free else "D"), 0.4)
                self.recover_keys = {{"A": pygame.K_a, "D": pygame.K_d, "S": pygame.K_s}[key]}
                self.recover_until = s.time + secs
                self.blocked_since = None
                self.recoveries += 1
                return self.recover_keys
        else:
            self.blocked_since = None
        x, y, yaw = s.sim.true_pose()
        while len(self.points) > 1 and math.dist((x, y), self.points[0]) < 0.12:  # no corner cutting
            self.points.pop(0)
        tx, ty = self.points[0]
        err = math.atan2(ty - y, tx - x) - yaw
        err = math.atan2(math.sin(err), math.cos(err))
        turn = pygame.K_a if err > 0 else pygame.K_d
        if abs(err) > 0.35:
            return {turn}  # turn in place first, like a careful driver
        if abs(err) > 0.12:
            return {pygame.K_w, turn}
        return {pygame.K_w}


def drive(seed: int, speed_level: int = C.DEFAULT_SPEED_LEVEL) -> dict:
    bot = KeyBot()
    folder = OUT / f"human_drive_level{speed_level + 1}"
    app = App(seed, screenshot=folder / f"human_drive_{seed}.png", frames=int((C.EPISODE_TIME_LIMIT + 2) * 60),
              key_policy=bot, speed_level=speed_level)
    app.frames_left = None
    original = app.update_episode

    def stop_when_done():
        original()
        if app.episode.status != "running" and app.frames_left is None:
            app.frames_left = 30  # keep the result banner on screen for half a second
    app.update_episode = stop_when_done
    result = app.run()
    result["recoveries"] = bot.recoveries
    return result


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    import argparse
    p = argparse.ArgumentParser(description="Scripted key drive of the held-out goals in the real window.")
    p.add_argument("seeds", type=int, nargs="*", help="goal seeds (default: the held-out seeds)")
    p.add_argument("--speed-level", type=int, default=C.DEFAULT_SPEED_LEVEL + 1,
                   choices=range(1, len(C.SPEED_LEVELS) + 1), help="speed level to drive at")
    a = p.parse_args()
    seeds = a.seeds or list(C.HELDOUT_SEEDS)
    folder = OUT / f"human_drive_level{a.speed_level}"
    folder.mkdir(parents=True, exist_ok=True)
    lines = [f"command: python tools/qa_human_drive.py {' '.join(sys.argv[1:])}",
             f"time: {time.strftime('%Y-%m-%d %H:%M:%S')}",
             f"speed level {a.speed_level}: {C.SPEED_LEVELS[a.speed_level - 1]}",
             f"seeds: {seeds}"]
    for line in lines:
        print(line)
    results = []
    for seed in seeds:
        r = drive(seed, a.speed_level - 1)
        results.append(r)
        line = (f"seed {seed}: {r['status']:8s} time {r['episode_time']:5.1f} s  collisions {r['collisions']}  "
                f"interventions {r['interventions']}  recoveries {r['recoveries']}  fps {r['fps']:.0f}  "
                f"rtf {r['real_time_factor']:.2f}")
        lines.append(line)
        print(line, flush=True)
    ok = all(r["status"] == "success" and r["collisions"] == 0 for r in results)
    lines.append(("ALL PASS" if ok else "FAILURES") + f"  (exit status {0 if ok else 1})")
    print(lines[-1])
    (folder / "results.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    sys.exit(0 if ok else 1)
