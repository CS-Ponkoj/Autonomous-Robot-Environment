"""Pygame window for manual driving: 3D view, robot camera, lidar map, and status.

Single thread, single event loop. Physics advances in fixed steps by the real time
that passed (capped), so a slow frame slows the simulation instead of skipping it. The window
(robot_env/display.py) is paced on the app's own 60 Hz schedule (no refresh wait), can be
resized, maximized, or made fullscreen (F11), and draws at a capped resolution the GPU scales.
"""

from __future__ import annotations

import argparse
import gc
import math
import time
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
import pygame

from . import config as C
from .layout import RoomMap, Task
from .cats import DEFAULT_CATS, MAX_CATS
from .display import Display
from .manual import ManualDriver, ManualInput
from .safety import is_safe
from .sim import CAMERA_LOOK, close_renderer
from .system import RobotSystem
from .types import Command, Driver

WINDOW = (1280, 720)  # initial window size; the window can be resized
FPS = 60
MAX_FRAME_DT = 0.05
SHADOW_LIGHTS = 3  # room lights casting shadows in the chase, orbit, and robot-camera views
# The top view shows the whole floor: shadows from the 2 room lights nearest the robot only (from
# above a shadow is a small dark patch, and each shadow light re-draws the whole scene, about 1 ms)
TOP_SHADOW_LIGHTS = 2
INSET_PERIOD = 1 / 10  # s of simulated time between robot-camera inset renders
VIEWS = ("chase", "top", "orbit", "robot camera")
HUD_MODES = ("compact", "full", "none")
HELP = [
    "W/A/S/D or arrows: drive    +/-: speed level    Shift: fastest level",
    "Right-drag: drive with the mouse",
    "Space: emergency brake (release Space, keys and mouse, then press a drive key)",
    "Left-drag: rotate view    Wheel: zoom    C: change view",
    "R: restart    N: next goal    T: continue after contact on/off",
    "Options button or O: options (number of cats)    F11: fullscreen    F12: screenshot    H: panels    Esc: quit",
]
SCREENSHOT_DIR = Path(__file__).resolve().parent.parent / "screenshots"


@dataclass
class MenuOption:
    """One line of the options menu: its name, the choices, the one in use, what choosing one
    does, and a note shown beside it."""
    name: str
    choices: tuple
    current: object  # () -> the choice in use
    apply: object  # (choice) -> None
    note: str = ""


class OptionsMenu:
    """The options panel, opened by the Options button on the panels (or O) and docked beside them,
    so the view and driving carry on. Click a choice to use it. More options are added as
    MenuOption lines."""

    def __init__(self, options: list[MenuOption]):
        self.button = pygame.Rect(16, 80, 96, 28)  # below the status lines (App places it)
        self.options = options
        self.open = False
        self._targets: list[tuple[pygame.Rect, int, object]] = []  # clickable choices (last drawn)
        self.panel = pygame.Rect(0, 0, 0, 0)

    def toggle(self) -> None:
        self.open = not self.open

    def choose(self, i: int, value) -> None:
        if value != self.options[i].current():
            self.options[i].apply(value)

    def handle_click(self, pos) -> bool:
        """True if the click was the menu's (the button or the open panel)."""
        if self.button.collidepoint(pos):
            self.toggle()
            return True
        if not self.open:
            return False
        for rect, i, value in self._targets:
            if rect.collidepoint(pos):
                self.choose(i, value)
                return True
        return self.panel.collidepoint(pos)

    def draw_button(self, screen: pygame.Surface, font) -> None:
        b = self.button
        pygame.draw.rect(screen, (60, 90, 130) if self.open else (30, 36, 48), b, border_radius=5)
        pygame.draw.rect(screen, (235, 240, 245), b, 1, border_radius=5)
        label = font.render("Options", True, (235, 240, 245))
        screen.blit(label, label.get_rect(center=b.center))

    def draw(self, screen: pygame.Surface, font) -> None:
        x, y = self.button.left, self.button.bottom + 8
        self.panel = pygame.Rect(x, y, 370, 52 + 58 * len(self.options))
        shade = pygame.Surface(self.panel.size, pygame.SRCALPHA)
        shade.fill((10, 14, 22, 225))
        screen.blit(shade, self.panel.topleft)
        pygame.draw.rect(screen, (235, 240, 245), self.panel, 1)
        self._targets = []
        y += 12
        for i, opt in enumerate(self.options):
            screen.blit(font.render(opt.name, True, (255, 205, 0)), (x + 14, y + 6))
            cx = x + 70
            for value in opt.choices:
                cell = pygame.Rect(cx, y, 40, 28)
                in_use = value == opt.current()
                pygame.draw.rect(screen, (60, 90, 130) if in_use else (40, 46, 58), cell, border_radius=4)
                if in_use:
                    pygame.draw.rect(screen, (90, 220, 140), cell, 2, border_radius=4)
                label = font.render(str(value), True, (235, 240, 245))
                screen.blit(label, label.get_rect(center=cell.center))
                self._targets.append((cell, i, value))
                cx += 48
            if opt.note:
                screen.blit(font.render(opt.note, True, (170, 180, 195)), (x + 14, y + 34))
            y += 58
        screen.blit(font.render("Click to choose.  O or Esc: close", True, (170, 180, 195)), (x + 14, self.panel.bottom - 26))


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


_SEARCHING = object()  # ViewCamera._pick: the search for a new pose is still running


class ViewCamera:
    """The viewer's camera. The chase and orbit views follow a SMOOTHED desired pose (so a
    small heading wobble never swings the view), but clearance is a hard limit applied every
    frame after smoothing: the camera never sits inside walls, furniture, or visual detail.

    Line of sight: the robot's top corners and a point ahead of it (cut short at the first
    obstacle) must be visible. When furniture hides them, the camera swings sideways and/or
    looks down more steeply to the nearest clear pose (closest to the current pose first),
    holds that choice for at least HOLD seconds (no left-right flip-flopping), and eases back
    to the normal pose once it is clear.

    Distance: three separate values. The wheel sets the REQUESTED distance (always, even when
    walls limit the view). A smoothed LIMIT follows the room available around the current and
    upcoming camera directions, so the camera pulls in gently before a wall would cut it short,
    and grows back only past a hysteresis margin. The DISPLAYED distance is the smaller of the
    two, then hard-capped by the clearance right now (the camera is never inside geometry)."""

    YAW_TAU = 0.3  # s, heading follow
    LOOK_TAU = 0.08  # s, look-at follow
    OUT_TAU = 0.25  # s, easing back out after an obstruction clears
    ELEV_TAU = 0.25  # s, tilting toward the elevation that keeps the robot in view
    OFFSET_TAU = 0.35  # s, swinging to or from a line-of-sight detour
    MARGIN = 0.12  # m kept between the camera and the first surface along its line of sight
    BUBBLE = 0.05  # m kept clear around the camera in every direction (near plane)
    HOLD = 0.6  # s a chosen detour is kept before reconsidering
    BLOCK_GRACE = 0.25  # s a blocked pose is kept (inside HOLD) before switching anyway
    OFFSETS = (0.0, 20.0, -20.0, 40.0, -40.0, 60.0, -60.0)  # degrees of sideways detour
    DETOUR_ELEVATIONS = (None, -45.0, -65.0, -80.0)  # None: the normal elevation; -80: over the robot
    ZOOM_STEP = 0.88  # distance factor per wheel notch toward the robot (12 percent)
    ZOOM_RATE = 45.0  # 1/s: critically damped follow, 90 percent of a notch in 86 ms, no overshoot
    LIMIT_IN_TAU = 0.10  # s, pulling in ahead of an obstruction
    LIMIT_HYST = 0.05  # m, the limit grows back only when the room exceeds it by this much
    LOOKAHEAD = (-16.0, -8.0, 8.0, 16.0)  # degrees around the azimuth the camera turns to, checked for room
    BEAM = ((0.35, 0.0), (-0.35, 0.0), (0.24, 0.0), (-0.24, 0.0), (0.12, 0.0), (-0.12, 0.0), (0.06, 0.0),
            (-0.06, 0.0), (0.0, 0.15), (0.0, 0.3), (0.0, -0.06))  # m (sideways, up):
    # parallel room rays, so an edge (a door jamb) is seen well before the line of sight reaches it
    LIMIT_IN_RATE = 5.5  # m/s at most when pulling in (a smooth glide, under 0.1 m per 60 Hz frame)
    LIMIT_OUT_RATE = 2.5  # m/s at most when easing back out
    LIMITED_CUE = 0.15  # m below the requested distance: show "zoom limited by walls"
    MAX_SPEED = 3.5  # m/s the camera point may move (about 6 cm per 60 Hz frame)
    SEARCH_PER_FRAME = 6  # candidate poses checked per frame while looking for a new one
    MIN_VIEW = 0.60  # m: closer than this the view loses the route around the robot
    CLOSEST = 0.40  # m: an edge crossing the line of sight may pull the camera this close, smoothly
    _BEAM_SIDE = np.array([b[0] for b in BEAM])
    _BEAM_LIFT = np.array([b[1] for b in BEAM])
    _BEAM_REACH = np.hypot(_BEAM_SIDE, _BEAM_LIFT)
    TAKE_ROOM = 0.90  # m of room a new pose must offer (hysteresis against flip-flopping)
    FOVY = (45.0, 70.0)  # degrees: normal field of view, and the widest used when the camera is close
    WIDE_BELOW = 1.2  # m: closer than this the view widens (keeps the robot and its surroundings in frame)
    ELEV_RATE = 40.0  # degrees/s the camera may tilt
    CORNERS = ((0.12, 0.08), (0.12, -0.08), (-0.12, 0.08), (-0.12, -0.08))  # robot frame, at z 0.16

    AZIMUTH, ELEVATION, DISTANCE = 0.0, -22.0, 1.6  # the default view (the user's angles and zoom)
    PITCH = (-85.0, -3.0)  # degrees: the user's tilt range, and the steepest automatic tilt
    MAX_DISTANCE = 6.0  # m: the longest zoom request
    TILT_BACK = 0.90  # m clear at the user's pitch before an automatic steeper tilt eases back (= TAKE_ROOM)
    USER_HOLD = 0.3  # s after the last drag during which no new detour is chosen
    FOV_TAU = 0.15  # s, widening or narrowing the lens
    DRAG_GAIN = 0.3  # degrees per pixel of left-drag
    TILT_CUE = (10.0, 0.3)  # degrees steeper than asked, for this many seconds: show "tilt limited by walls"

    def __init__(self) -> None:
        self.cam = mujoco.MjvCamera()
        self.mode = 0
        self.reset_user()
        self.reset()

    def reset_user(self) -> None:
        """Back to the default angles and zoom (a new episode)."""
        self.azimuth, self.elevation, self.distance = self.AZIMUTH, self.ELEVATION, self.DISTANCE

    def reset(self) -> None:
        """Forget the smoothed state (episode reset, teleport, view change): snap next frame."""
        self._init = False
        self._heading = 0.0
        self._look = np.zeros(3)
        self._want = self.distance  # smoothed requested distance
        self._want_rate = 0.0  # its rate of change (log distance per second)
        self._limit = math.inf  # smoothed room available
        self._growing = False
        self.room = math.inf
        self._search = None  # an unfinished search for a new pose (see _pick)
        self._settled_prev = {}  # last settled elevation per pose (search starts there next frame)
        self._search_at = 0
        self.fovy = self.FOVY[0]  # field of view for the main view (degrees)
        self._shown = None  # last displayed (azimuth, elevation, distance, look-at)
        self._blocked_for = 0.0
        self.limited = False  # walls keep the camera closer than requested: the window shows a cue
        self._elev = self.elevation
        self._offset = 0.0  # smoothed sideways detour (degrees)
        self._detour = (0.0, None)  # chosen (offset, elevation override)
        self._hold_until = -1.0
        self._clock = 0.0
        self.blocked = False  # no clear pose found: the window shows a small cue
        self.tilt_limited = False  # walls keep the camera steeper than the user's pitch: a cue
        self._steep_for = 0.0
        self._user_az = self._user_el = 0.0  # the user's drag since the last frame (degrees)
        self._held = False  # the left button is down on the view (a drag in progress)
        self._user_time = -math.inf  # clock time of the last drag motion
        self._tilt_dir = 0.0  # direction of the drag's tilt (+1 up, -1 down) while it lasts
        self.ignore_drag = False  # the press went to a panel or menu: its drag does not turn the view
        self._targets = []

    def handle_event(self, event: pygame.event.Event) -> None:
        """Left-drag turns and tilts the view (applied directly, the same frame); the wheel zooms."""
        t = event.type
        third_person = VIEWS[self.mode] in ("chase", "orbit")  # the other views ignore the mouse
        if t == pygame.MOUSEBUTTONDOWN and event.button == 1:
            self._held = not self.ignore_drag
        elif t == pygame.MOUSEBUTTONUP and event.button == 1:
            self._held, self.ignore_drag, self._tilt_dir = False, False, 0.0
        elif t == pygame.MOUSEMOTION and event.buttons[0] and not self.ignore_drag and third_person:
            self._held = True
            daz = -event.rel[0] * self.DRAG_GAIN
            el = float(np.clip(self.elevation - event.rel[1] * self.DRAG_GAIN, *self.PITCH))
            self.azimuth += daz
            self._user_az += daz
            self._user_el += el - self.elevation
            if event.rel[1]:
                self._tilt_dir = -1.0 if event.rel[1] > 0 else 1.0
            self.elevation = el
            if event.rel[0] or event.rel[1]:
                self._user_time = self._clock
        elif (t == pygame.MOUSEMOTION and not event.buttons[0]) or t == pygame.WINDOWFOCUSLOST:
            # the press is over (a release the window never saw)
            self._held, self.ignore_drag, self._tilt_dir = False, False, 0.0
        elif t == pygame.MOUSEWHEEL and third_person:
            if event.y > 0 and self._init and self._limit < self.distance:
                # Walls hold the camera in: zoom in from where it is, so the first notch shows.
                shown = max(min(self._want, self._limit), 0.5)
                self.distance = self._want = shown
                self._want_rate = 0.0
            self.distance = float(np.clip(self.distance * self.ZOOM_STEP ** event.y, 0.5, self.MAX_DISTANCE))

    @property
    def user_active(self) -> bool:
        """A drag is in progress or just ended: the user, not the detour search, picks the angle."""
        return self._held or self._clock - self._user_time < self.USER_HOLD

    @classmethod
    def _follow_zoom(cls, shown: float, rate: float, target: float, dt: float) -> tuple[float, float]:
        """One frame of the zoom: a critically damped spring on the log of the distance (every
        notch feels the same near and far), integrated exactly, so it is the same at any frame
        rate and never overshoots. A notch the other way first stops the old motion, so the view
        never keeps drifting the wrong way."""
        err = math.log(shown) - math.log(target)
        if err * rate > 0.0:  # moving away from the (new) target: stop first
            rate = 0.0
        w, t = cls.ZOOM_RATE, max(dt, 0.0)
        decay = math.exp(-w * t)
        err_new = (err + (rate + w * err) * t) * decay
        rate_new = (rate - w * (rate + w * err) * t) * decay
        return target * math.exp(err_new), rate_new

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

        self._clock += max(dt, 0.0)
        # ray results for this frame (the look-at moves)
        self._clear_cache, self._beam_cache, self._sight_cache = {}, {}, {}
        user_az, user_el = self._user_az, self._user_el  # the user's drag since the last frame
        # The tilt direction of a drag in progress, kept between its motion events (a mouse may
        # report less often than frames are drawn) until the release or USER_HOLD without motion.
        tilting = self._held and self._clock - self._user_time < self.USER_HOLD
        drag_dir = self._tilt_dir if tilting else 0.0
        self._user_az = self._user_el = 0.0
        heading = math.degrees(yaw) if view == "chase" else 0.0
        target_look = np.array([x, y, 0.1])
        snap = not self._init
        if snap:
            self._heading, self._look = heading, target_look.copy()
            self._want, self._elev = self.distance, self.elevation
            self._init = True
        else:
            wrap = (heading - self._heading + 180.0) % 360.0 - 180.0  # shortest way round
            self._heading += wrap * self._alpha(dt, self.YAW_TAU)
            self._look += (target_look - self._look) * self._alpha(dt, self.LOOK_TAU)
        self.cam.lookat[:] = self._look
        targets = self._targets = self._sight_targets(sim, x, y, yaw)

        # Line-of-sight detour, spring-arm style: a pose that hides the robot first pulls the camera
        # in along its own line (in front of whatever hides it); a pose is kept while that leaves
        # MIN_VIEW. Only then the camera swings or tilts to the nearest pose offering TAKE_ROOM, and
        # only when none exists does it accept a close-up down to CLOSEST. Every switch respects the
        # hold, except: a blocked pose is left after BLOCK_GRACE, a cramped one at once. While the
        # user drags (and USER_HOLD after), the hand picks the angle: a detour is cancelled, no new
        # one is chosen, and only the arm shortens.
        normal = (0.0, None)
        if self.user_active and (user_az or user_el) and self._detour != normal:
            self._detour, self._hold_until, self._search = normal, self._clock + self.HOLD, None
        sight = self._pose_sight(sim, *self._detour, targets)
        good = sight is not None and sight >= min(self.MIN_VIEW, self.distance) - 1e-6
        if snap:
            self.blocked = False
            if not good:
                # first frame: search everything now
                choice = self._pick(sim, targets, budget=None, fallback=sight is None)
                self._detour = choice or self._detour
                self.blocked = choice is None and sight is None
        elif good:
            self._blocked_for = 0.0
            self._search = None  # no switch needed any more
            self.blocked = False
            if self._detour != normal and self._clock >= self._hold_until and not self.user_active:
                # back to normal once it offers TAKE_ROOM, or at least what is shown now (the
                # switch never pulls the camera in)
                back = self._pose_sight(sim, *normal, targets)
                need = min(self.TAKE_ROOM, self.distance, max(self.cam.distance, self.MIN_VIEW))
                if back is not None and back >= need - 1e-6:
                    self._detour, self._hold_until = normal, self._clock + self.HOLD
        elif self.user_active:
            self._search = None
            self.blocked = sight is None
        else:
            self._blocked_for += max(dt, 0.0)
            cramped = sight is not None  # sees the robot only from closer than MIN_VIEW: leave now
            if cramped or self._clock >= self._hold_until or self._blocked_for >= self.BLOCK_GRACE:
                choice = self._pick(sim, targets, fallback=not cramped)
                if choice is _SEARCHING:
                    pass  # the search continues next frame; keep the current pose meanwhile
                elif choice is not None and choice != self._detour:
                    self._detour, self._hold_until = choice, self._clock + self.HOLD
                    self._blocked_for = 0.0
                if choice is not _SEARCHING:
                    self.blocked = choice is None and not cramped
        offset, elev_override = self._detour
        if snap:
            self._offset = offset  # first frame after a reset: start at the chosen pose
        else:
            # The user's tilt shows at once where the camera is at the user's pitch; a camera tilted
            # steeper by the walls stays there until the user's pitch is the steeper one.
            e0 = self._elev
            if abs(e0 - (self.elevation - user_el)) < 0.5:
                self._elev = self.elevation
            else:
                self._elev = min(e0, self.elevation)
            user_el = self._elev - e0
        prev_offset, prev_elev = self._offset, self._elev
        self._offset += (offset - self._offset) * self._alpha(dt, self.OFFSET_TAU)
        self.cam.azimuth = self._heading + self.azimuth + self._offset

        # Elevation: the user's pitch (or the detour's), tilted steeper automatically only when even
        # MIN_VIEW is not clear there (and eased back only past TILT_BACK, so it never flip-flops).
        target_elev = self._settled_elevation(sim, self.cam.azimuth, elev_override)
        if snap:
            self._elev = target_elev  # first frame after a reset: start at a clear angle
        step = (target_elev - self._elev) * self._alpha(dt, self.ELEV_TAU)
        step = float(np.clip(step, -self.ELEV_RATE * max(dt, 0.0), self.ELEV_RATE * max(dt, 0.0)))
        if step * drag_dir < 0:
            step = 0.0  # while the user tilts, the automatic tilt never moves the other way
        self._elev += step
        if not snap and self.cam.distance > 0:
            # Swing and tilt only as fast as the distance can glide: if the new angle would cut the
            # camera short by more than one frame's glide, keep the old angle while it pulls in.
            reach = self.cam.distance - self.LIMIT_IN_RATE * max(dt, 0.0)
            new_az = self.cam.azimuth
            if self._clear_distance(sim, new_az, self._elev) < reach - 1e-3:
                self._offset, self._elev = prev_offset, prev_elev
                self.cam.azimuth = self._heading + self.azimuth + self._offset
        self.cam.elevation = self._elev

        # Distance: requested (wheel), smoothed limit (room around the current and upcoming
        # directions), and the hard clearance cap right now (applied last, every frame).
        az, el = self.cam.azimuth, self.cam.elevation
        was_free = self._limit >= self._want - 1e-6  # the walls did not hold the camera in last frame
        if snap:
            self._want, self._want_rate = self.distance, 0.0
        else:
            self._want, self._want_rate = self._follow_zoom(self._want, self._want_rate, self.distance, dt)
        clear = self._clear_distance(sim, az, el)
        hard = self._bubble_distance(sim, az, el, min(self._want, clear))  # never inside geometry, now
        hard_room = hard if self.distance <= self._want else self._bubble_distance(sim, az, el, min(self.distance, clear))
        # Sight: the arm is shortened (never swung) to stay in front of whatever would hide the robot
        # (at once: the robot is never hidden) or the way ahead (smoothly, through the limit).
        robot = targets[:len(self.CORNERS)]
        hard = min(hard, self._sight_at(sim, az, el, robot, hard) or hard)
        hard_room = min(hard_room, self._sight_at(sim, az, el, targets, hard_room) or hard_room)
        # Room around where the camera is and where it is heading (its target side and tilt).
        heading_az = heading + self.azimuth + offset  # where the camera is turning to (unsmoothed heading)
        near = self._beam_room(sim, az, el)  # edges about to cross the line of sight as the robot moves
        goal_clear = self._clear_distance(sim, heading_az, target_elev)
        ahead = min([self._beam_room(sim, az, target_elev), goal_clear, self._beam_room(sim, heading_az, target_elev),
                     self._bubble_distance(sim, heading_az, target_elev, min(goal_clear, self.distance))]
                    + [self._clear_distance(sim, az, el + d) for d in (-6.0, -3.0, 3.0)]
                    + [self._clear_distance(sim, heading_az + d, e) for d in self.LOOKAHEAD for e in (el, target_elev)])
        # Directions the camera is not facing yet pull it in early, but never below MIN_VIEW: a door
        # frame beside the robot is not a reason to collapse the view (if the line of sight itself
        # gets that tight, the pose is switched for a roomier one).
        room = min(hard_room, max(near, self.CLOSEST), max(ahead, self.MIN_VIEW))
        room = min(room, 2.0 * self.distance)  # beyond this the limit does not matter
        self.room = room  # what the walls allow around the current and upcoming pose (diagnostics, cue)
        if snap or self._limit == math.inf or (was_free and room >= self._limit):
            # nothing to ease: the limit is not what the camera shows (the zoom spring is), so it
            # follows the room at once and never slows the user's zoom
            self._limit = room
        elif room < self._limit:
            self._growing = False
            step = (room - self._limit) * self._alpha(dt, self.LIMIT_IN_TAU)
            self._limit += max(step, -self.LIMIT_IN_RATE * dt)
        elif room > self._limit + self.LIMIT_HYST or (self._growing and room > self._limit + 0.005):
            # hysteresis on entry only: once easing out, go all the way back to the room
            self._growing = True
            step = (room - self._limit) * self._alpha(dt, self.OUT_TAU)
            self._limit += min(step, self.LIMIT_OUT_RATE * dt)
        else:
            self._growing = False
        self.cam.distance = min(self._want, self._limit, hard)
        if not snap and self._shown is not None:
            self._limit_step(sim, dt, user_az, user_el)
        # Walls hold the camera close: widen the view smoothly (a real camera operator would use a
        # wider lens). A distance the user chose keeps the normal lens, so zooming in magnifies.
        lo, hi = self.FOVY
        t = float(np.clip((min(self.distance, self.WIDE_BELOW) - self.cam.distance) / (self.WIDE_BELOW - self.MIN_VIEW),
                          0.0, 1.0))
        wide = lo + (hi - lo) * t
        self.fovy = wide if snap else self.fovy + (wide - self.fovy) * self._alpha(dt, self.FOV_TAU)
        self._shown = (self.cam.azimuth, self.cam.elevation, self.cam.distance, np.array(self.cam.lookat))
        self.limited = self.distance - self._limit > self.LIMITED_CUE  # the smoothed limit: no flicker
        steep = self.cam.elevation < self.elevation - self.TILT_CUE[0]
        self._steep_for = self._steep_for + max(dt, 0.0) if steep else 0.0
        self.tilt_limited = self._steep_for >= self.TILT_CUE[1]
        return self.cam

    def _limit_step(self, sim, dt: float, user_az: float = 0.0, user_el: float = 0.0) -> None:
        """Cap how far the camera moves on its own in one frame (MAX_SPEED): if the new pose is
        further, show the point part of the way there (angles and distance blended) and let the
        filters continue from it. The user's own turn and tilt this frame are never slowed: the
        step is measured from last frame's pose turned by them. Clearance and sight still win: the
        blended pose is capped at its own angle, so a sudden obstacle can still pull the camera in."""
        az0, el0, d0, look0 = self._shown
        az0, el0 = az0 + user_az, el0 + user_el
        p0 = _camera_point(look0, az0, el0, d0)
        az1, el1, d1 = self.cam.azimuth, self.cam.elevation, self.cam.distance
        look1 = np.array(self.cam.lookat)
        limit = self.MAX_SPEED * max(dt, 1e-6)
        # The user's own zoom (the camera showing the requested distance) is never slowed: only
        # the turning and the look-at count against the cap then.
        zooming = abs(d1 - self._want) < 1e-9
        if np.linalg.norm(_camera_point(look1, az1, el1, d0 if zooming else d1) - p0) <= limit:
            return
        daz = (az1 - az0 + 180.0) % 360.0 - 180.0
        lo, hi = 0.0, 1.0
        for _ in range(12):  # largest blend whose camera point stays within the step
            f = (lo + hi) / 2
            q = _camera_point(look1, az0 + daz * f, el0 + (el1 - el0) * f, d0 if zooming else d0 + (d1 - d0) * f)
            lo, hi = (f, hi) if np.linalg.norm(q - p0) <= limit else (lo, f)
        az, el = az0 + daz * lo, el0 + (el1 - el0) * lo
        dist = d1 if zooming else d0 + (d1 - d0) * lo
        dist = min(dist, self._clear_distance(sim, az, el))
        dist = self._bubble_distance(sim, az, el, dist)
        dist = min(dist, self._sight_at(sim, az, el, self._targets[:len(self.CORNERS)], dist) or dist)
        self.cam.azimuth, self.cam.elevation, self.cam.distance = az, el, dist
        self._elev = el  # the filters continue from what is shown
        self._offset = az - (self._heading + self.azimuth)

    def _bubble_distance(self, sim, azimuth: float, elevation: float, dist: float) -> float:
        """Shorten `dist` until a BUBBLE-sized neighbourhood around the camera is clear (the line
        of sight can run along a wall, leaving the camera too close to it sideways)."""
        az, el = math.radians(azimuth), math.radians(elevation)
        back = np.array([-math.cos(el) * math.cos(az), -math.cos(el) * math.sin(az), -math.sin(el)])
        right = np.cross(-back, (0.0, 0.0, 1.0))
        right /= max(np.linalg.norm(right), 1e-9)
        up = np.cross(right, -back)
        probes = (right, -right, up, -up, back)
        while dist > 0.02:
            pos = np.asarray(self.cam.lookat) + back * dist
            if all(not (0 <= sim.ray_to_view_blocker(pos, d) <= self.BUBBLE) for d in probes):
                return dist
            dist -= 0.03
        return 0.02

    # ----- line of sight -----
    def _sight_targets(self, sim, x: float, y: float, yaw: float) -> list:
        """The robot's top corners and a point ahead of it, cut short at the first obstacle."""
        c, s = math.cos(yaw), math.sin(yaw)
        pts = [np.array([x + c * px - s * py, y + s * px + c * py, 0.16]) for px, py in self.CORNERS]
        origin = np.array([x, y, 0.16])
        forward = np.array([c, s, 0.0])
        ahead = sim.ray_to_view_blocker(origin, forward)
        reach = 1.0 if ahead < 0 else max(min(1.0, ahead - 0.2), 0.0)
        pts.append(origin + forward * reach)
        return pts

    def _settled_elevation(self, sim, azimuth: float, elev_override) -> float:
        """The elevation the camera settles at: the user's pitch (or the detour's), unless even
        MIN_VIEW is not clear there; then the shallowest steeper angle (down to PITCH[0]) at which
        it is. A steeper tilt eases back toward the user's pitch only where TILT_BACK is clear
        (hysteresis: driving along a wall never flips between two pitches). The zoom request
        above MIN_VIEW plays no part: zooming never tilts the view."""
        start = self.elevation if elev_override is None else min(self.elevation, elev_override)
        keep, take = min(self.distance, self.MIN_VIEW), min(max(self.distance, self.MIN_VIEW), self.TILT_BACK)
        floor = self.PITCH[0]

        def ok(el, need):
            return self._clear_distance(sim, azimuth, el) >= need and self._beam_room(sim, azimuth, el) >= need
        # Start from last frame's answer for this pose (the scene changes little between frames):
        # walk up while the next shallower angle is clear, or down until one is.
        key = (round(azimuth - self._heading - self.azimuth, 1), elev_override)
        el = self._settled_prev.get(key)
        if el is None or el > start:
            el = start
        el = max(min(el, start), floor)
        if self._held and ok(start, keep):
            el = start  # the user is tilting: their pitch only needs MIN_VIEW (TILT_BACK is for the automatic return)
        elif ok(el, keep):
            while el + 2.0 <= start and ok(el + 2.0, take):
                el += 2.0
            if el + 2.0 > start and el < start and ok(start, take):
                el = start
        else:
            while el > floor and not ok(el, keep):
                el -= 2.0
            el = max(el, floor)
        self._settled_prev[key] = el
        return float(el)

    def _sight_at(self, sim, azimuth: float, elevation: float, targets, dist: float) -> float | None:
        """The longest arm (within one step), at most `dist`, from which the camera sees `targets`
        (spring arm: step in along the line of sight to just in front of whatever hides a target,
        and check again, since an edge can hide the robot from one distance and not from another).
        None when not even CLOSEST works from this direction."""
        key = (round(azimuth, 2), round(elevation, 2), round(dist, 3), len(targets))
        cache = getattr(self, "_sight_cache", None)
        if cache is not None and key in cache:
            return cache[key]
        az, el = math.radians(azimuth), math.radians(elevation)
        back = np.array([-math.cos(el) * math.cos(az), -math.cos(el) * math.sin(az), -math.sin(el)])
        look = np.asarray(self.cam.lookat, dtype=float)
        d, result = dist, None
        for step in range(13):
            if d < self.CLOSEST - 1e-9:
                break
            if step == 12:
                d = self.CLOSEST  # out of steps: the last try is the closest allowed view
            pos = look + back * d
            blocker = None
            for p in targets:
                v = p - pos
                n = float(np.linalg.norm(v))
                if n < 1e-6:
                    continue
                hit = sim.ray_to_view_blocker(pos, v / n)
                if 0 <= hit < n - 0.03:
                    blocker = pos + v / n * hit
                    break
            if blocker is None:
                result = d
                break
            if d <= self.CLOSEST + 1e-9:
                break  # not even the closest allowed view sees the robot
            # just in front of it (steps of at least 5 cm or 20 percent: 12 of them reach CLOSEST from
            # the longest arm, 6 x 0.8^12 = 0.41 m, so the answer is within one step of the longest),
            # never skipping CLOSEST
            d = max(min(d - max(0.05, 0.2 * d), float(np.dot(blocker - look, back)) - 0.05), self.CLOSEST)
        if cache is not None:
            cache[key] = result
        return result

    def _pose_sight(self, sim, offset: float, elev_override, targets) -> float | None:
        """_sight_at for a pose judged where it settles: its angle, its settled tilt, and the arm
        the requested distance and the walls allow (side room never alone rules a pose out)."""
        azimuth = self._heading + self.azimuth + offset
        elevation = self._settled_elevation(sim, azimuth, elev_override)  # judged where it will settle
        room = min(self.distance, self._clear_distance(sim, azimuth, elevation),
                   max(self._beam_room(sim, azimuth, elevation), self.CLOSEST))
        return self._sight_at(sim, azimuth, elevation, targets, room)

    def _pick(self, sim, targets, budget: int | None = SEARCH_PER_FRAME, fallback: bool = True):
        """A usable new pose (sees the robot and offers TAKE_ROOM), else, with `fallback`, any pose
        that sees it from at least CLOSEST; None when nothing qualifies. The search checks at most
        `budget` candidates per frame (it continues next frame and returns _SEARCHING meanwhile),
        so it never stalls a frame."""
        if self._search is None:
            order = self._candidates()
            self._search = [(pose, self.TAKE_ROOM) for pose in order]
            if fallback:
                self._search += [(pose, self.CLOSEST) for pose in order]
            self._search_at = 0
        left = len(self._search) if budget is None else budget
        while left > 0 and self._search_at < len(self._search):
            pose, room = self._search[self._search_at]
            self._search_at += 1
            left -= 1
            sight = self._pose_sight(sim, *pose, targets)
            if sight is not None and sight >= min(room, self.distance) - 1e-6:
                self._search = None
                return pose
        if self._search_at >= len(self._search):
            self._search = None
            return None
        return _SEARCHING

    def _candidates(self) -> list:
        """Every (offset, elevation) pose, closest to the current one first (continuity, then the
        smallest detour; the current side wins ties)."""
        cur_offset, cur_elev = self._detour
        levels = list(self.DETOUR_ELEVATIONS)
        cur_level = levels.index(cur_elev)
        side = 1.0 if cur_offset >= 0 else -1.0

        def cost(pose):
            offset, elev = pose
            return (abs(offset - cur_offset) / 20.0 + abs(levels.index(elev) - cur_level) + abs(offset) / 40.0
                    + levels.index(elev) * 0.5, offset * side < 0)
        return sorted(((o, e) for e in levels for o in self.OFFSETS), key=cost)

    def _beam_room(self, sim, azimuth: float, elevation: float) -> float:
        """Room along rays parallel to the line of sight, offset sideways and up: an edge (a door
        frame) is seen before the line of sight itself reaches it, so the camera glides in early."""
        key = (round(azimuth, 2), round(elevation, 2))
        cache = getattr(self, "_beam_cache", None)
        if cache is not None and key in cache:
            return cache[key]
        az, el = math.radians(azimuth), math.radians(elevation)
        ca, sa, ce, se = math.cos(az), math.sin(az), math.cos(el), math.sin(el)
        back = np.array([-ce * ca, -ce * sa, -se])
        right = np.array([sa, -ca, 0.0])  # horizontal, perpendicular to the line of sight
        up = np.array([-se * ca, -se * sa, ce])  # perpendicular to both (camera up)
        offsets = self._BEAM_SIDE[:, None] * right + self._BEAM_LIFT[:, None] * up
        look = np.asarray(self.cam.lookat, dtype=float)
        # one batched cast from the look-at: does each parallel ray start in open space?
        gaps = sim.rays_to_view_blocker(look, offsets / self._BEAM_REACH[:, None], self._BEAM_REACH.max() + 0.06)
        best = math.inf
        for i in range(len(offsets)):
            if 0 <= gaps[i] <= self._BEAM_REACH[i] + 0.05:
                continue  # this ray would start inside or behind a wall next to the robot: not room ahead
            hit = sim.ray_to_view_blocker(look + offsets[i], back)
            if hit >= 0:
                best = min(best, max(hit - self.MARGIN, 0.02))
        if cache is not None:
            cache[key] = best
        return best

    def _clear_distance(self, sim, azimuth: float, elevation: float) -> float:
        """Distance the camera may sit behind the look-at point without entering geometry
        (solids and visual detail, so it never sits inside a chair's arms)."""
        key = (round(azimuth, 2), round(elevation, 2))
        cache = getattr(self, "_clear_cache", None)
        if cache is not None and key in cache:
            return cache[key]
        az, el = math.radians(azimuth), math.radians(elevation)
        back = np.array([-math.cos(el) * math.cos(az), -math.cos(el) * math.sin(az), -math.sin(el)])
        hit = sim.ray_to_view_blocker(self.cam.lookat, back)
        value = math.inf if hit < 0 else max(hit - self.MARGIN, 0.02)
        if cache is not None:
            cache[key] = value
        return value


@dataclass
class Episode:
    task: Task
    status: str = "running"  # running, success, collision, timeout
    collisions_seen: int = 0
    end_time: float | None = None  # simulated time when the episode ended


class App:
    def __init__(self, seed: int, screenshot: Path | None = None, frames: int | None = None,
                 script: list[ScriptStep] | None = None, view: int = 0, driver: Driver | None = None,
                 key_policy=None, speed_level: int = C.DEFAULT_SPEED_LEVEL, cats: int = 0, cat_seed: int = 0):
        # Only display and fonts. pygame.init() also starts the joystick subsystem,
        # and MuJoCo's GLFW renderer then triggers a JOYDEVICEREMOVED event that
        # crashes pygame.event.get() with KeyError(0) (found in the step 2 spike).
        pygame.display.init()
        pygame.font.init()
        pygame.event.set_blocked([pygame.JOYDEVICEADDED, pygame.JOYDEVICEREMOVED])
        self.display = Display(WINDOW, "Autonomous Robot Environment")
        self.font = pygame.font.SysFont("consolas", 16)
        self.big = pygame.font.SysFont("consolas", 30, bold=True)
        self.clock = pygame.time.Clock()
        self.cat_count, self.cat_seed = cats, cat_seed
        self.system = RobotSystem(cats=cats, cat_seed=cat_seed)
        self.room = RoomMap(self.system.sim.model)
        self.menu = OptionsMenu([
            MenuOption("Cats", tuple(range(MAX_CATS + 1)), lambda: self.cat_count, self.set_cats,
                       "changing it restarts the simulation"),
        ])
        self.renderer = None
        self._make_renderer()
        self._inset = None  # cached robot-camera inset and the simulated time it shows
        self._inset_pending = None  # (size, time) of an inset scene prepared, drawn next frame
        self._inset_pixels = None
        self._light_view = None  # the view the lights were last chosen for (a change: no fade)
        self._inset_time = -math.inf
        self.frame_times: list[float] = []  # seconds between presented frames (measured)
        self.view = ViewCamera()
        self.overview_option = mujoco.MjvOption()  # default groups: ceiling (3) hidden
        self.view.mode = view
        self._frame_dt = 1 / FPS
        self.input = ManualInput(speed_level)
        # Any driver uses the same decide(observation) plug. The human can always brake
        # (Space) and focus loss always stops; release-to-stop applies only to manual driving.
        self.driver: Driver = driver or ManualDriver(self.input)
        self.continue_after_contact = False
        self.hud = "compact"  # H cycles: compact (driving), full (diagnostics), none
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

    @property
    def screen(self) -> pygame.Surface:
        """The frame being drawn (render resolution; the GPU scales it to the window)."""
        return self.display.surface

    def _make_renderer(self) -> None:
        w, h = self.screen.get_size()
        if self.renderer is not None:
            close_renderer(self.renderer)
        self.renderer = mujoco.Renderer(self.system.sim.model, height=h, width=w)

    @property
    def show_help(self) -> bool:
        """Panels shown at all (compact or full)."""
        return self.hud != "none"

    @show_help.setter
    def show_help(self, on: bool) -> None:
        self.hud = "full" if on else "none"

    # ----- episodes -----
    def set_cats(self, n: int) -> None:
        """Restart with n cats (the cats are part of the world model, so the world is rebuilt);
        the same goal seed, view, and speed level."""
        if not 0 <= n <= MAX_CATS:
            raise ValueError(f"cats must be 0 to {MAX_CATS}")
        close_renderer(self.renderer)  # bound to the old model
        self.renderer = None
        self.system.close()
        self.cat_count = n
        self.system = RobotSystem(cats=n, cat_seed=self.cat_seed)
        self.room = RoomMap(self.system.sim.model)
        self._light_view = None
        self._make_renderer()
        self.new_episode(self.seed)

    def new_episode(self, seed: int) -> None:
        self.seed = seed
        task = self.room.sample_task(seed)
        x, y, yaw = task.start
        self.system.reset(x, y, yaw, task.goal)
        reset = getattr(self.driver, "reset", None) if hasattr(self, "driver") else None
        if callable(reset):
            reset()  # a stateful driver (map, odometry, clock) starts the new episode fresh
        self.input.release_all()  # drive inputs only: a latched or held brake stays on
        self.episode = Episode(task)
        if hasattr(self, "view"):
            self.view.reset()  # the robot teleported: snap the camera
        self._drop_inset()  # nothing from before the reset is shown
        self._next_decision = 0.0
        self._last_sent = None

    def _drop_inset(self) -> None:
        """Forget the robot-camera preview (shown, and any scene prepared for it): after a reset,
        while it is not on screen, or when its size changes, it starts again from the present."""
        self._inset = self._inset_pixels = self._inset_pending = None
        self._inset_time = -math.inf

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
            if s.cats is not None:
                s.finish_cat_contacts("episode_end")
            if s.log is not None:
                s.log.event("episode_" + ep.status, s.time, seed=self.seed)

    # ----- input -----
    def handle_events(self) -> bool:
        for event in pygame.event.get():
            if event.type in (pygame.QUIT, pygame.WINDOWCLOSE):
                return False
            if self.display.handle_event(event):
                self._make_renderer()
            if (event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and self.hud != "none"
                    and self.menu.handle_click(self._surface_pos(event.pos))):
                self.view.ignore_drag = True  # dragging on from the menu does not turn the view
                continue  # the Options button or its panel (not a drag of the view)
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    if self.menu.open:
                        self.menu.toggle()  # Esc closes the options first
                        continue
                    return False
                if event.key == pygame.K_o:
                    if self.hud == "none":
                        self.hud = "compact"  # the options sit beside the panels: show them
                    self.menu.toggle()
                    continue
                if event.key == pygame.K_c:
                    self.view.mode = (self.view.mode + 1) % len(VIEWS)
                    self.view.reset()
                elif event.key == pygame.K_r:
                    self.view.reset_user()  # a restart also restores the default view
                    self.new_episode(self.seed)
                elif event.key == pygame.K_n:
                    self.view.reset_user()
                    self.new_episode(self.seed + 1)
                elif event.key == pygame.K_t:
                    self.continue_after_contact = not self.continue_after_contact
                elif event.key == pygame.K_h:
                    self.hud = HUD_MODES[(HUD_MODES.index(self.hud) + 1) % len(HUD_MODES)]
                    if self.hud == "none":
                        self.menu.open = False  # hidden with the panels
                elif event.key == pygame.K_F12:
                    self.save_screenshot()
            self.view.handle_event(event)
            self.input.handle_event(event)
        return True

    def _surface_pos(self, pos) -> tuple[int, int]:
        """A window position in drawing coordinates (the frame is drawn at render resolution and
        scaled to the window, so above the render cap the two differ)."""
        (ww, wh), (sw, sh) = self.display.window_size, self.screen.get_size()
        return int(pos[0] * sw / max(ww, 1)), int(pos[1] * sh / max(wh, 1))

    # ----- simulation -----
    def simulate(self, real_dt: float) -> None:
        s = self.system
        manual = self.manual_driving
        s.flags.manual_mode = manual
        s.flags.manual_input_held = self.input.drive_input_held if manual else False
        s.flags.emergency_brake = self.input.emergency_brake
        s.flags.focus_lost = not self.input.focused
        if self.episode.status != "running" and not s.flags.episode_over:
            s.end_episode()  # an ended episode always holds the stop (reset clears it)
        remaining = min(real_dt, MAX_FRAME_DT)
        self.sim_elapsed += remaining
        running = self.episode.status == "running"
        if running and manual and self.input.drive_input_held and self.input.command() != self._last_sent:
            self._decide()  # fresh input: send right away instead of waiting for the next tick
            self._next_decision = s.time + C.DECISION_PERIOD
        while remaining > 1e-9:
            if not running:
                # Ended: physics runs on (the robot is stopped), but no driver is asked again.
                s.advance(remaining)
                break
            if s.time >= self._next_decision - 1e-9:
                self._decide()
                self._next_decision = s.time + C.DECISION_PERIOD
            chunk = min(remaining, max(self._next_decision - s.time, C.PHYSICS_DT))
            s.advance(chunk)
            remaining -= chunk
            self.update_episode()
            if self.episode.status != "running":
                s.end_episode()  # the robot stops now, not at the next control tick
                running = False

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
        self._draw_scene()
        if self.hud != "none":
            self.menu.button.top = 80 if self.hud == "compact" else 220  # under the status lines
            self.menu.draw_button(self.screen, self.font)
            if self.menu.open:
                self.menu.draw(self.screen, self.font)

    def _draw_scene(self) -> None:
        s = self.system
        camera = self.view.apply(s.sim, self._frame_dt)  # viewer only
        # The robot's own camera shows the ceiling; overview cameras hide it (cutaway).
        option = s.sim.camera_option if camera == "robot_cam" else self.overview_option
        vis = s.sim.model.vis.global_
        normal_fovy = vis.fovy
        if VIEWS[self.view.mode] in ("chase", "orbit"):
            vis.fovy = self.view.fovy  # free-camera field of view (the robot camera has its own)
        s.sim.before_render()  # pose the visual-only skins (cats) for this frame
        self._choose_lights(camera)
        self.renderer.update_scene(s.sim.data, camera=camera, scene_option=option)
        # the top view looks straight down on the whole floor: its faint floor reflections (4-6%)
        # are invisible from there, and the reflection pass re-draws the scene
        self.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = VIEWS[self.view.mode] != "top"
        # a faint haze far off, in the views from inside the rooms (from above it would grey the floor)
        self.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_FOG] = VIEWS[self.view.mode] != "top"
        vis.fovy = normal_fovy
        image = self.renderer.render()
        self.screen.blit(_surface(image), (0, 0))

        if self.hud == "none":
            self._drop_inset()  # not on screen: a later preview starts from the present
            _text_box(self.screen, self.font, ["H: show panels"], (24, 20))
            self.draw_banners()
            return
        width, height = self.screen.get_size()
        compact = self.hud == "compact"
        if camera == "robot_cam":  # the main view already shows the robot camera: no duplicate inset
            self._drop_inset()
        else:
            cam_rect = pygame.Rect(width - 256, 16, 240, 180) if compact else pygame.Rect(width - 336, 16, 320, 240)
            size = (cam_rect.height, cam_rect.width)
            if (self._inset is not None and self._inset.get_size() != cam_rect.size) or \
                    (self._inset_pending is not None and self._inset_pending[0] != size):
                self._drop_inset()  # the panels changed size: a preview at the new size, from now
            if self._inset_pending is not None:
                # second half: draw the scene prepared last frame (one frame old, on a 10 Hz preview)
                image = s.sim.render_prepared()
                self._inset_pixels = image  # the cached surface shares these pixels
                self._inset, self._inset_time = _surface(image), self._inset_pending[1]
                self._inset_pending = None
            else:
                if self._inset is None or abs(s.time - self._inset_time) >= INSET_PERIOD - 1e-9:
                    # first half: the scene, at the shown size, without shadow or floor-reflection
                    # passes (each re-draws the whole scene, and the small inset shows neither);
                    # the skins were posed for this frame. Drawn in the next frame: the cost is
                    # split over two frames.
                    m = s.sim.model
                    cast = m.light_castshadow.copy()
                    m.light_castshadow[:] = 0
                    try:
                        s.sim.prepare_camera(size, posed=True, reflections=False, shadows=False)
                    finally:
                        m.light_castshadow[:] = cast
                    self._inset_pending = (size, s.time)
            if self._inset is not None:
                self.screen.blit(self._inset, cam_rect.topleft)
            pygame.draw.rect(self.screen, (235, 240, 245), cam_rect, 2)
            _text_box(self.screen, self.font, ["robot camera"], (cam_rect.x + 8, cam_rect.bottom + 9))
        lidar = 160 if compact else 220
        self.draw_lidar(pygame.Rect(width - lidar - 16, height - lidar - 16, lidar, lidar))

        obs = s.observe()
        res = s.last_result
        if compact:  # driving essentials only, so the view stays clear (H: full diagnostics)
            safety = ", ".join(res.reasons) if res.reasons else "ok"
            if "clearance" in res.reasons:
                safety += f"   free moves: {self.free_moves_text()}"
            _text_box(self.screen, self.font, [
                f"{self.speed_level_text()}   goal {obs.goal_distance:4.2f} m at {math.degrees(obs.goal_bearing):+4.0f} deg"
                f"   time {self.episode_time():5.1f}/{C.EPISODE_TIME_LIMIT:.0f} s",
                f"safety: {safety}   collisions {s.collisions}   {self.clock.get_fps():3.0f} FPS",
            ], (24, 20))
            _text_box(self.screen, self.font, ["H: more    C: view    O: options    F11: fullscreen    Esc: quit"],
                      (24, height - 36))
            self.draw_banners()
            return
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
        _text_box(self.screen, self.font, HELP, (24, height - 20 * len(HELP) - 16))
        self.draw_banners()

    def _choose_lights(self, camera) -> None:
        """The lights drawn (the renderer draws only eight): those of the room the camera is in,
        faded over a change so nothing pops; the top view, which shows every room, a fixed set.
        Shadows from the SHADOW_LIGHTS drawn room lights nearest the robot (the rooms in view), or
        the TOP_SHADOW_LIGHTS nearest it from above. Each shadow-casting light re-draws the whole scene
        (about 1 ms each), so this is the main rendering cost."""
        sim = self.system.sim
        m = sim.model
        top = VIEWS[self.view.mode] == "top"
        rx, ry, ryaw = sim.true_pose()
        look = (rx + CAMERA_LOOK * math.cos(ryaw), ry + CAMERA_LOOK * math.sin(ryaw))  # the robot camera's
        if camera == "robot_cam" or top:
            x, y = rx, ry
        else:  # where the free camera's eye is
            az, el = math.radians(camera.azimuth), math.radians(camera.elevation)
            x = camera.lookat[0] - camera.distance * math.cos(el) * math.cos(az)
            y = camera.lookat[1] - camera.distance * math.cos(el) * math.sin(az)
            look = (float(camera.lookat[0]), float(camera.lookat[1]))
        if self._light_view != self.view.mode:
            sim.lights.snap(x, y, top, look)
            self._light_view = self.view.mode
        else:
            sim.lights.fade(x, y, self._frame_dt, top, look)
        sim.lights.shadows(rx, ry, TOP_SHADOW_LIGHTS if top else SHADOW_LIGHTS)

    def draw_banners(self) -> None:
        obs = self.system.observe()
        if self.input.emergency_brake:
            _banner(self.screen, self.big, "EMERGENCY BRAKE - " + self.input.brake_hint(), (240, 170, 40), y=200)
        if self.input.pending_release and not self.input.emergency_brake:
            _banner(self.screen, self.font, "Release " + ", ".join(self.input.pending_names())
                    + " and press again to drive", (240, 200, 90), y=200)
        if not self.input.focused:
            _banner(self.screen, self.big, "WINDOW NOT FOCUSED - STOPPED", (240, 170, 40), y=255)
        if VIEWS[self.view.mode] in ("chase", "orbit"):
            if self.view.blocked:
                _text_box(self.screen, self.font, ["view blocked by furniture: press C for another view"], (24, 300))
            elif self.view.limited or self.view.tilt_limited:
                what = " and ".join(n for n, on in (("zoom", self.view.limited), ("tilt", self.view.tilt_limited)) if on)
                _text_box(self.screen, self.font, [f"{what} limited by walls"], (24, 300))
        status = self.episode.status
        if status != "running":
            text = {"success": "GOAL REACHED!", "collision": "COLLISION - episode over",
                    "timeout": "TIME LIMIT - episode over"}[status]
            color = (60, 200, 90) if status == "success" else (230, 70, 60)
            _banner(self.screen, self.big, text + "   (R: restart, N: next goal)", color,
                    y=self.screen.get_height() // 2 - 20)
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
        _text_box(self.screen, self.font, ["lidar"], (rect.x, rect.y - 25))  # forward is up (the arrow)

    def save_screenshot(self, path: Path | None = None) -> Path:
        path = path or SCREENSHOT_DIR / f"screenshot_{time.strftime('%Y%m%d_%H%M%S')}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        pygame.image.save(self.screen, str(path))
        return path

    # ----- main loop -----
    def run(self) -> dict:
        # Everything built so far (world, renderer, cats) lives for the whole run: moved out of the
        # garbage collector's view, a full collection no longer walks it (a 20+ ms frame). Only
        # when the caller has frozen nothing itself (its own freeze is never undone), and undone
        # however the run ends; the window and renderers are closed however it ends, too.
        froze = gc.isenabled() and gc.get_freeze_count() == 0
        if froze:
            gc.collect()
            gc.freeze()
        try:
            try:
                summary = self._run_loop()
            except BaseException:
                self._close_all()  # the run's own error is the one raised
                raise
            failed = self._close_all()
            if failed is not None:
                raise failed
            return summary
        finally:
            if froze:
                gc.unfreeze()  # collectable again (a process may run more than one App)

    def _close_all(self) -> Exception | None:
        """Close the main renderer, the system (with the robot camera and log), the window, and
        pygame, each one even if an earlier one fails; returns the first failure (or None)."""
        failed = None
        for close in (lambda: close_renderer(self.renderer), self.system.close, self.display.close, pygame.quit):
            try:
                close()
            except Exception as e:  # keep closing the rest
                failed = failed or e
        return failed

    def _run_loop(self) -> dict:
        running = True
        last = time.perf_counter()
        self._last_present = None
        self._next_slot = None
        while running:
            self._advance_script()
            running = self.handle_events()
            now = time.perf_counter()
            real_dt, last = now - last, now
            self.wall_elapsed += real_dt  # true wall time; sim time is capped per frame
            self.simulate(real_dt)
            self._frame_dt = min(real_dt, MAX_FRAME_DT)
            self.draw()
            # Even application pacing: each frame is handed over at its slot of a fixed 1/FPS
            # schedule (resynced after a late frame). Not synchronized to the display refresh, so
            # a display at another rate or phase can still repeat or skip a refresh.
            self._next_slot = present_deadline(self._next_slot, time.perf_counter())
            _wait_until(self._next_slot)
            self.display.present()  # no vsync wait; the window system shows the frame
            self.clock.tick()
            done = time.perf_counter()
            if self._last_present is not None:
                self.frame_times.append(done - self._last_present)
            self._last_present = done
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


def _camera_point(look, azimuth: float, elevation: float, distance: float) -> np.ndarray:
    az, el = math.radians(azimuth), math.radians(elevation)
    back = np.array([-math.cos(el) * math.cos(az), -math.cos(el) * math.sin(az), -math.sin(el)])
    return np.asarray(look, dtype=float) + back * distance


def present_deadline(slot: float | None, now: float, fps: float = FPS) -> float:
    """The next frame's slot on a fixed 1/fps schedule. A frame that is already late (the slot
    passed by more than one period) starts a new schedule from now instead of rushing to catch up."""
    period = 1.0 / fps
    if slot is None or now > slot + period:
        return now
    return slot + period


def _wait_until(t: float, clock=time.perf_counter, sleep=time.sleep) -> None:
    """Sleep coarsely, then spin for the last 2 ms (Windows sleeps are coarse)."""
    while True:
        left = t - clock()
        if left <= 0:
            return
        if left > 0.002:
            sleep(left - 0.002)


def _surface(image: np.ndarray) -> pygame.Surface:
    """A surface showing the RGB image, sharing its pixels (no copy): the caller keeps the array
    alive as long as the surface is used."""
    h, w, _ = image.shape
    return pygame.image.frombuffer(np.ascontiguousarray(image, dtype=np.uint8), (w, h), "RGB")


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
    x = (screen.get_width() - box.get_width()) // 2
    screen.blit(box, (x, y))
    screen.blit(surf, (x + 15, y + 8))


def main(argv: list[str] | None = None) -> dict:
    p = argparse.ArgumentParser(description="Drive the robot by hand.")
    p.add_argument("--seed", type=int, default=C.HELDOUT_SEEDS[0], help="goal seed (start and goal positions)")
    p.add_argument("--view", type=int, default=0, help="0 chase, 1 top, 2 orbit, 3 robot camera")
    p.add_argument("--cats", type=int, default=DEFAULT_CATS, choices=range(0, MAX_CATS + 1),
                   help=f"number of wandering cats, 0 to {MAX_CATS} (default {DEFAULT_CATS}; 0 turns them off)")
    p.add_argument("--cat-seed", type=int, default=0, help="seed for the cats' behavior (independent of the goal)")
    p.add_argument("--speed-level", type=int, default=C.DEFAULT_SPEED_LEVEL + 1,
                   choices=range(1, len(C.SPEED_LEVELS) + 1),
                   help=f"starting speed level, 1 to {len(C.SPEED_LEVELS)} (default {C.DEFAULT_SPEED_LEVEL + 1})")
    p.add_argument("--frames", type=int, help="quit after this many frames (automated checks)")
    p.add_argument("--screenshot", type=Path, help="save a screenshot here when quitting")
    p.add_argument("--script", help="scripted key holds, e.g. 'W:2;W+A:1;none:0.5' (automated checks)")
    a = p.parse_args(argv)
    if a.cat_seed < 0:
        p.error("--cat-seed must be 0 or more")
    app = App(a.seed, a.screenshot, a.frames, parse_script(a.script) if a.script else None, a.view,
              speed_level=a.speed_level - 1, cats=a.cats, cat_seed=a.cat_seed)
    summary = app.run()
    print("summary:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in summary.items()})
    return summary
