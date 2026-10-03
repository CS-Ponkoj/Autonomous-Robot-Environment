"""Pygame window for manual driving: 3D view, robot camera, lidar map, and status.

Single thread, single event loop. Physics advances in fixed steps by the real time
that passed (capped), so a slow frame slows the simulation instead of skipping it.
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
import pygame

from . import config as C
from .layout import RoomMap, Task
from .manual import ManualDriver, ManualInput
from .safety import is_safe
from .system import RobotSystem
from .types import Command, Driver

WINDOW = (1280, 720)
FPS = 60
MAX_FRAME_DT = 0.05
VIEWS = ("chase", "top", "orbit", "robot camera")
HELP = [
    "W/A/S/D or arrows: drive    +/-: speed level    Shift: fastest level",
    "Right-drag: drive with the mouse",
    "Space: emergency brake (release Space, keys and mouse, then press a drive key)",
    "Left-drag: rotate view    Wheel: zoom    C: change view",
    "R: restart    N: next goal    T: continue after contact on/off",
    "F12: screenshot    H: hide/show panels    Esc: quit",
]
SCREENSHOT_DIR = Path(__file__).resolve().parent.parent / "screenshots"


@dataclass
class ScriptStep:
    keys: frozenset[int]
    duration: float


def parse_script(text: str) -> list[ScriptStep]:
    """'W:2;W+A:1;none:0.5;SPACE:0.3' -> held keys and how long (seconds)."""
    names = {"W": pygame.K_w, "A": pygame.K_a, "S": pygame.K_s, "D": pygame.K_d,
             "UP": pygame.K_UP, "DOWN": pygame.K_DOWN, "LEFT": pygame.K_LEFT, "RIGHT": pygame.K_RIGHT,
             "SHIFT": pygame.K_LSHIFT, "SPACE": pygame.K_SPACE,
             "C": pygame.K_c, "R": pygame.K_r, "N": pygame.K_n, "H": pygame.K_h, "T": pygame.K_t}
    steps = []
    for part in filter(None, (p.strip() for p in text.split(";"))):
        keys_text, dur = part.rsplit(":", 1)
        keys = frozenset() if keys_text.strip().lower() == "none" else frozenset(
            names[k.strip().upper()] for k in keys_text.split("+"))
        steps.append(ScriptStep(keys, float(dur)))
    return steps


class ViewCamera:
    """The viewer's camera. The chase and orbit views follow a SMOOTHED desired pose (so a
    small heading wobble never swings the view), but wall/furniture clearance is a hard
    limit applied every frame after smoothing: the camera never sits inside geometry."""

    YAW_TAU = 0.3  # s, heading follow
    LOOK_TAU = 0.08  # s, look-at follow
    OUT_TAU = 0.4  # s, easing back out after an obstruction clears
    ELEV_TAU = 0.25  # s, tilting toward the elevation that keeps the robot in view
    MARGIN = 0.12  # m kept between the camera and the first solid surface

    def __init__(self) -> None:
        self.cam = mujoco.MjvCamera()
        self.mode = 0
        self.azimuth = 0.0
        self.elevation = -22.0
        self.distance = 1.6
        self.reset()

    def reset(self) -> None:
        """Forget the smoothed state (episode reset, teleport, view change): snap next frame."""
        self._init = False
        self._heading = 0.0
        self._look = np.zeros(3)
        self._dist = self.distance
        self._elev = self.elevation

    def handle_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.MOUSEMOTION and event.buttons[0]:
            self.azimuth -= event.rel[0] * 0.3
            self.elevation = float(np.clip(self.elevation - event.rel[1] * 0.3, -89.0, -3.0))
        elif event.type == pygame.MOUSEWHEEL:
            self.distance = float(np.clip(self.distance * 0.9 ** event.y, 0.5, 10.0))

    @staticmethod
    def _alpha(dt: float, tau: float) -> float:
        return 1.0 - math.exp(-max(dt, 0.0) / tau)  # frame-rate independent

    def apply(self, sim, dt: float = 1 / 60) -> mujoco.MjvCamera | str:
        """Set up the viewer camera for the current mode (uses the true pose: viewer only)."""
        x, y, yaw = sim.true_pose()
        view = VIEWS[self.mode]
        if view == "robot camera":
            return "robot_cam"
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        if view == "top":
            self.cam.lookat[:] = (0.0, 0.0, 0.0)
            self.cam.azimuth, self.cam.elevation = 90.0, -89.9
            # Fit the whole floor; the wall tops are 2.4 m closer to the camera than the floor.
            self.cam.distance = (C.FLOOR_HALF_SIZE + 0.3) / math.tan(math.radians(22.5)) + C.WALL_HEIGHT
            return self.cam

        heading = math.degrees(yaw) if view == "chase" else 0.0
        target_look = np.array([x, y, 0.1])
        if not self._init:
            self._heading, self._look = heading, target_look.copy()
            self._dist, self._elev = self.distance, self.elevation
            self._init = True
        else:
            wrap = (heading - self._heading + 180.0) % 360.0 - 180.0  # shortest way round
            self._heading += wrap * self._alpha(dt, self.YAW_TAU)
            self._look += (target_look - self._look) * self._alpha(dt, self.LOOK_TAU)
        self.cam.lookat[:] = self._look
        self.cam.azimuth = self._heading + self.azimuth

        # Elevation: tilt smoothly toward the shallowest angle (from the user's choice down to
        # -85 degrees) at which the desired distance is clear.
        wanted = min(self.distance, 0.9)
        target_elev = -85.0
        for el in np.arange(self.elevation, -85.0, -1.0):
            if self._clear_distance(sim, self.cam.azimuth, el) >= wanted:
                target_elev = float(el)
                break
        self._elev += (target_elev - self._elev) * self._alpha(dt, self.ELEV_TAU)
        self.cam.elevation = self._elev

        # Distance: ease back out slowly, pull in immediately (hard cap, every frame).
        if self.distance > self._dist:
            self._dist += (self.distance - self._dist) * self._alpha(dt, self.OUT_TAU)
        else:
            self._dist = self.distance
        clear = self._clear_distance(sim, self.cam.azimuth, self.cam.elevation)
        self.cam.distance = min(self._dist, clear)
        self._dist = self.cam.distance if clear < self._dist else self._dist
        return self.cam

    def _clear_distance(self, sim, azimuth: float, elevation: float) -> float:
        """Distance the camera may sit behind the look-at point without entering geometry."""
        az, el = math.radians(azimuth), math.radians(elevation)
        back = np.array([-math.cos(el) * math.cos(az), -math.cos(el) * math.sin(az), -math.sin(el)])
        hit = sim.ray_to_solid(self.cam.lookat, back)
        if hit < 0:
            return math.inf
        return max(hit - self.MARGIN, 0.02)


@dataclass
class Episode:
    task: Task
    status: str = "running"  # running, success, collision, timeout
    collisions_seen: int = 0
    end_time: float | None = None  # simulated time when the episode ended


class App:
    def __init__(self, seed: int, screenshot: Path | None = None, frames: int | None = None,
                 script: list[ScriptStep] | None = None, view: int = 0, driver: Driver | None = None,
                 key_policy=None, speed_level: int = C.DEFAULT_SPEED_LEVEL):
        # Only display and fonts. pygame.init() also starts the joystick subsystem,
        # and MuJoCo's GLFW renderer then triggers a JOYDEVICEREMOVED event that
        # crashes pygame.event.get() with KeyError(0) (found in the step 2 spike).
        pygame.display.init()
        pygame.font.init()
        pygame.event.set_blocked([pygame.JOYDEVICEADDED, pygame.JOYDEVICEREMOVED])
        self.screen = pygame.display.set_mode(WINDOW)
        pygame.display.set_caption("Autonomous Robot Environment")
        self.font = pygame.font.SysFont("consolas", 16)
        self.big = pygame.font.SysFont("consolas", 30, bold=True)
        self.clock = pygame.time.Clock()
        self.system = RobotSystem()
        self.room = RoomMap(self.system.sim.model)
        self.renderer = mujoco.Renderer(self.system.sim.model, height=WINDOW[1], width=WINDOW[0])
        self.view = ViewCamera()
        self.overview_option = mujoco.MjvOption()  # default groups: ceiling (3) hidden
        self.view.mode = view
        self._frame_dt = 1 / FPS
        self.input = ManualInput(speed_level)
        # Any driver uses the same decide(observation) plug. The human can always brake
        # (Space) and focus loss always stops; release-to-stop applies only to manual driving.
        self.driver: Driver = driver or ManualDriver(self.input)
        self.continue_after_contact = False
        self.show_help = True
        self.seed = seed
        self.frames_left = frames
        self.screenshot = screenshot
        self.script = list(script or [])
        self._script_keys: frozenset[int] = frozenset()
        # QA only: a function (app) -> set of keys to hold, evaluated every frame. Keys are
        # posted as real pygame key events, so they take the same path as a person's keys.
        self.key_policy = key_policy
        self._script_until = 0.0
        self.wall_start = time.perf_counter()
        self.sim_elapsed = 0.0
        self.wall_elapsed = 0.0
        self.new_episode(seed)

    # ----- episodes -----
    def new_episode(self, seed: int) -> None:
        self.seed = seed
        task = self.room.sample_task(seed)
        x, y, yaw = task.start
        self.system.reset(x, y, yaw, task.goal)
        self.input.release_all()  # drive inputs only: a latched or held brake stays on
        self.episode = Episode(task)
        if hasattr(self, "view"):
            self.view.reset()  # the robot teleported: snap the camera
        self._next_decision = 0.0
        self._last_sent = None

    def update_episode(self) -> None:
        ep, s = self.episode, self.system
        if ep.status != "running":
            return
        obs = s.observe()
        new_collision = s.collisions > ep.collisions_seen
        ep.collisions_seen = s.collisions
        if new_collision and not self.continue_after_contact:
            ep.status = "collision"
        elif obs.goal_distance <= C.GOAL_RADIUS and not new_collision:
            ep.status = "success"
        elif s.time >= C.EPISODE_TIME_LIMIT:
            ep.status = "timeout"
        if ep.status != "running":
            ep.end_time = min(s.time, C.EPISODE_TIME_LIMIT)

    # ----- input -----
    def handle_events(self) -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                if event.key == pygame.K_c:
                    self.view.mode = (self.view.mode + 1) % len(VIEWS)
                    self.view.reset()
                elif event.key == pygame.K_r:
                    self.new_episode(self.seed)
                elif event.key == pygame.K_n:
                    self.new_episode(self.seed + 1)
                elif event.key == pygame.K_t:
                    self.continue_after_contact = not self.continue_after_contact
                elif event.key == pygame.K_h:
                    self.show_help = not self.show_help
                elif event.key == pygame.K_F12:
                    self.save_screenshot()
            self.view.handle_event(event)
            self.input.handle_event(event)
        return True

    # ----- simulation -----
    def simulate(self, real_dt: float) -> None:
        s = self.system
        manual = self.manual_driving
        s.flags.manual_mode = manual
        s.flags.manual_input_held = self.input.drive_input_held if manual else False
        s.flags.emergency_brake = self.input.emergency_brake
        s.flags.focus_lost = not self.input.focused
        s.flags.episode_over = self.episode.status != "running"  # robot brakes to a stop
        remaining = min(real_dt, MAX_FRAME_DT)
        self.sim_elapsed += remaining
        if manual and self.input.drive_input_held and self.input.command() != self._last_sent:
            self._decide()  # fresh input: send right away instead of waiting for the next tick
            self._next_decision = s.time + C.DECISION_PERIOD
        while remaining > 1e-9:
            if s.time >= self._next_decision - 1e-9:
                self._decide()
                self._next_decision = s.time + C.DECISION_PERIOD
            chunk = min(remaining, max(self._next_decision - s.time, C.PHYSICS_DT))
            s.advance(chunk)
            remaining -= chunk
            self.update_episode()
            s.flags.episode_over = self.episode.status != "running"

    @property
    def manual_driving(self) -> bool:
        return isinstance(self.driver, ManualDriver)

    def free_moves(self) -> list[str]:
        """Which single manual keys the clearance check would allow right now."""
        v, w = self.input.speed()
        options = [("W", Command(v, 0.0)), ("A", Command(0.0, w)), ("D", Command(0.0, -w)),
                   ("S", Command(-v * C.MANUAL_REVERSE_FACTOR, 0.0))]
        obs = self.system.observe()
        return [key for key, cmd in options if is_safe(cmd, obs)]

    def free_moves_text(self) -> str:
        names = {"W": "forward (W)", "A": "turn left (A)", "D": "turn right (D)", "S": "back up (S)"}
        free = self.free_moves()
        return ", ".join(names[k] for k in free) if free else "none (press Space, then R)"

    def speed_level_text(self) -> str:
        """The selected level, and the level in effect when Shift is held."""
        n = len(C.SPEED_LEVELS)
        v, w = C.SPEED_LEVELS[self.input.level]
        text = f"speed level {self.input.level + 1}/{n}: {v:.2f} m/s, {w:.1f} rad/s"
        if self.input.boosted:
            bv, bw = C.SPEED_LEVELS[-1]
            text += f"   Shift: level {n} ({bv:.2f} m/s, {bw:.1f} rad/s) in effect"
        return text

    def episode_time(self) -> float:
        """Episode clock; it stops when the episode ends (the physics keeps running)."""
        return self.episode.end_time if self.episode.end_time is not None else self.system.time

    def _decide(self) -> None:
        decision = self.driver.decide(self.system.observe())
        self.system.apply(decision)
        self._last_sent = decision.command if decision else None

    # ----- drawing -----
    def draw(self) -> None:
        s = self.system
        camera = self.view.apply(s.sim, self._frame_dt)  # viewer only
        # The robot's own camera shows the ceiling; overview cameras hide it (cutaway).
        option = s.sim.camera_option if camera == "robot_cam" else self.overview_option
        self.renderer.update_scene(s.sim.data, camera=camera, scene_option=option)
        self.screen.blit(_surface(self.renderer.render()), (0, 0))

        if not self.show_help:
            _text_box(self.screen, self.font, ["H: show panels"], (24, 20))
            self.draw_banners()
            return
        if camera != "robot_cam":  # the main view already shows the robot camera: no duplicate inset
            cam_rect = pygame.Rect(WINDOW[0] - 336, 16, 320, 240)
            self.screen.blit(_surface(s.sim.render_camera()), cam_rect.topleft)
            pygame.draw.rect(self.screen, (235, 240, 245), cam_rect, 2)
            _text_box(self.screen, self.font, ["robot camera"], (cam_rect.x + 8, cam_rect.bottom + 9))
        self.draw_lidar(pygame.Rect(WINDOW[0] - 236, WINDOW[1] - 236, 220, 220))

        obs = s.observe()
        res = s.last_result
        fresh = s.command_age is not None and s.command_age <= C.COMMAND_LIFETIME
        req = s.requested if (fresh and (self.input.drive_input_held or not self.manual_driving)) else None
        rtf = self.sim_elapsed / self.wall_elapsed if self.wall_elapsed > 0 else 0.0
        lines = [
            f"driver: {self.driver.name}   goal seed {self.seed}   view: {VIEWS[self.view.mode]}   time {self.episode_time():5.1f}/{C.EPISODE_TIME_LIMIT:.0f} s",
            f"goal: {obs.goal_distance:4.2f} m at {math.degrees(obs.goal_bearing):+4.0f} deg",
            self.speed_level_text(),
            f"speed (encoders): {obs.velocity_estimate[0]:+.2f} m/s  {obs.velocity_estimate[1]:+.2f} rad/s",
            f"requested: {_cmd(req)}   approved: {_cmd(res.command)}",
            f"applied to motors: {_cmd(s.applied)}",
            f"safety: {', '.join(res.reasons) if res.reasons else 'ok'}"
            + (f"   free moves: {self.free_moves_text()}" if "clearance" in res.reasons else ""),
            f"collisions {s.collisions}   interventions {s.intervention_events}   "
            f"contact-continue {'ON' if self.continue_after_contact else 'off'}",
            f"{self.clock.get_fps():4.0f} FPS   real-time factor {rtf:4.2f}",
        ]
        _text_box(self.screen, self.font, lines, (24, 20))
        _text_box(self.screen, self.font, HELP, (24, WINDOW[1] - 20 * len(HELP) - 16))
        self.draw_banners()

    def draw_banners(self) -> None:
        obs = self.system.observe()
        if self.input.emergency_brake:
            _banner(self.screen, self.big, "EMERGENCY BRAKE - " + self.input.brake_hint(), (240, 170, 40), y=200)
        if self.input.pending_release and not self.input.emergency_brake:
            _banner(self.screen, self.font, "Release " + ", ".join(self.input.pending_names())
                    + " and press again to drive", (240, 200, 90), y=200)
        if not self.input.focused:
            _banner(self.screen, self.big, "WINDOW NOT FOCUSED - STOPPED", (240, 170, 40), y=255)
        status = self.episode.status
        if status != "running":
            text = {"success": "GOAL REACHED!", "collision": "COLLISION - episode over",
                    "timeout": "TIME LIMIT - episode over"}[status]
            color = (60, 200, 90) if status == "success" else (230, 70, 60)
            _banner(self.screen, self.big, text + "   (R: restart, N: next goal)", color, y=WINDOW[1] // 2 - 20)
        if obs.contact:
            pygame.draw.rect(self.screen, (240, 60, 50), self.screen.get_rect(), 6)

    def draw_lidar(self, rect: pygame.Rect) -> None:
        obs = self.system.observe()
        panel = pygame.Surface(rect.size, pygame.SRCALPHA)
        panel.fill((10, 14, 22, 200))
        cx, cy = rect.width // 2, rect.height // 2
        scale = (rect.width / 2 - 10) / 3.0  # up to 3 m
        # One polygon for the free space and one outline for the returns (360 rays: drawing a
        # line and a dot per ray cost several milliseconds per frame).
        ok = obs.lidar_valid
        shown = np.where(ok, np.minimum(obs.lidar, 3.0), 0.4)
        px = cx - np.sin(obs.lidar_angles) * shown * scale
        py = cy - np.cos(obs.lidar_angles) * shown * scale
        points = list(zip(px.tolist(), py.tolist()))
        pygame.draw.polygon(panel, (40, 62, 86, 200), points)
        pygame.draw.lines(panel, (90, 220, 140), True, points, 2)
        for i in np.flatnonzero(~ok):
            pygame.draw.line(panel, (200, 120, 40), (cx, cy), points[i], 1)
        for i in np.flatnonzero(ok & (obs.lidar < 0.5)):
            pygame.draw.circle(panel, (240, 70, 60), (int(px[i]), int(py[i])), 3)
        gb = obs.goal_bearing
        gd = min(obs.goal_distance, 3.0)
        pygame.draw.circle(panel, (60, 230, 100), (int(cx - math.sin(gb) * gd * scale), int(cy - math.cos(gb) * gd * scale)), 5, 2)
        hl, hw = C.FOOTPRINT_HALF_LENGTH * scale, C.FOOTPRINT_HALF_WIDTH * scale
        pygame.draw.rect(panel, (255, 205, 0), pygame.Rect(cx - hw, cy - hl, 2 * hw, 2 * hl), 1)
        pygame.draw.polygon(panel, (255, 205, 0), [(cx, cy - hl), (cx - 4, cy - hl + 7), (cx + 4, cy - hl + 7)])
        self.screen.blit(panel, rect.topleft)
        _text_box(self.screen, self.font, ["lidar (up = forward)"], (rect.x + 8, rect.y - 25))

    def save_screenshot(self, path: Path | None = None) -> Path:
        path = path or SCREENSHOT_DIR / f"screenshot_{time.strftime('%Y%m%d_%H%M%S')}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        pygame.image.save(self.screen, str(path))
        return path

    # ----- main loop -----
    def run(self) -> dict:
        running = True
        last = time.perf_counter()
        while running:
            self._advance_script()
            running = self.handle_events()
            now = time.perf_counter()
            real_dt, last = now - last, now
            self.wall_elapsed += real_dt  # true wall time; sim time is capped per frame
            self.simulate(real_dt)
            self._frame_dt = min(real_dt, MAX_FRAME_DT)
            self.draw()
            pygame.display.flip()
            self.clock.tick(FPS)
            if self.frames_left is not None:
                self.frames_left -= 1
                if self.frames_left <= 0:
                    running = False
        if self.screenshot:
            self.save_screenshot(self.screenshot)
        summary = {
            "status": self.episode.status, "seed": self.seed, "sim_time": self.system.time,
            "episode_time": self.episode_time(),
            "collisions": self.system.collisions, "interventions": self.system.intervention_events,
            "goal_distance": self.system.observe().goal_distance, "fps": self.clock.get_fps(),
            "real_time_factor": self.sim_elapsed / self.wall_elapsed if self.wall_elapsed else 0.0,
        }
        self.renderer.close()
        self.system.close()
        pygame.quit()
        return summary

    def _advance_script(self) -> None:
        """Replay scripted key holds (for automated QA that drives like a human)."""
        if self.key_policy is not None:
            self._post_keys(frozenset(self.key_policy(self)))
            return
        if not self.script and not self._script_keys:
            return
        now = time.perf_counter() - self.wall_start
        if now < self._script_until:
            return
        step = self.script.pop(0) if self.script else ScriptStep(frozenset(), 0.0)
        self._post_keys(step.keys)
        self._script_until = now + step.duration

    def _post_keys(self, keys: frozenset[int]) -> None:
        for k in self._script_keys - keys:
            pygame.event.post(pygame.event.Event(pygame.KEYUP, key=k, mod=0, unicode="", scancode=0))
        for k in keys - self._script_keys:
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=k, mod=0, unicode="", scancode=0))
        self._script_keys = keys


def _surface(image: np.ndarray) -> pygame.Surface:
    h, w, _ = image.shape
    return pygame.image.frombuffer(image.tobytes(), (w, h), "RGB")


def _cmd(c) -> str:
    return "none" if c is None else f"{c.v:+.2f} m/s {c.omega:+.2f} rad/s"


def _text_box(screen, font, lines, pos) -> None:
    x, y = pos
    box = pygame.Surface((max(font.size(t)[0] for t in lines) + 16, len(lines) * 20 + 10), pygame.SRCALPHA)
    box.fill((10, 14, 22, 175))
    screen.blit(box, (x - 8, y - 5))
    for line in lines:
        screen.blit(font.render(line, True, (235, 240, 245)), (x, y))
        y += 20


def _banner(screen, font, text, color, y) -> None:
    surf = font.render(text, True, color)
    box = pygame.Surface((surf.get_width() + 30, surf.get_height() + 16), pygame.SRCALPHA)
    box.fill((10, 14, 22, 210))
    x = (WINDOW[0] - box.get_width()) // 2
    screen.blit(box, (x, y))
    screen.blit(surf, (x + 15, y + 8))


def main(argv: list[str] | None = None) -> dict:
    p = argparse.ArgumentParser(description="Drive the robot by hand.")
    p.add_argument("--seed", type=int, default=C.HELDOUT_SEEDS[0], help="goal seed (start and goal positions)")
    p.add_argument("--view", type=int, default=0, help="0 chase, 1 top, 2 orbit, 3 robot camera")
    p.add_argument("--speed-level", type=int, default=C.DEFAULT_SPEED_LEVEL + 1,
                   choices=range(1, len(C.SPEED_LEVELS) + 1),
                   help=f"starting speed level, 1 to {len(C.SPEED_LEVELS)} (default {C.DEFAULT_SPEED_LEVEL + 1})")
    p.add_argument("--frames", type=int, help="quit after this many frames (automated checks)")
    p.add_argument("--screenshot", type=Path, help="save a screenshot here when quitting")
    p.add_argument("--script", help="scripted key holds, e.g. 'W:2;W+A:1;none:0.5' (automated checks)")
    a = p.parse_args(argv)
    app = App(a.seed, a.screenshot, a.frames, parse_script(a.script) if a.script else None, a.view,
              speed_level=a.speed_level - 1)
    summary = app.run()
    print("summary:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in summary.items()})
    return summary
