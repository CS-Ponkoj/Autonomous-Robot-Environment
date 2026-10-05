"""GUI smoke tests: the real window, renderer, and event loop, driven by scripted keys.

These show the window works end to end. They do not replace the owner's manual drive."""

import numpy as np
import pygame
import pytest

from robot_env.app import VIEWS, App, parse_script
from robot_env.sim import close_renderer


def run_app(tmp_path, **kw):
    shot = tmp_path / "shot.png"
    app = App(kw.pop("seed", 1000), screenshot=shot, **kw)
    summary = app.run()
    return summary, shot


@pytest.mark.gui
def test_window_renders_and_saves_screenshot(tmp_path):
    summary, shot = run_app(tmp_path, frames=20)
    assert shot.exists() and shot.stat().st_size > 10_000
    assert summary["status"] == "running" and summary["collisions"] == 0


@pytest.mark.gui
@pytest.mark.parametrize("view", [0, 1, 2, 3])
def test_every_view_renders(tmp_path, view):
    summary, shot = run_app(tmp_path, frames=5, view=view)
    assert shot.exists()


@pytest.mark.gui
@pytest.mark.parametrize("view, expect_inset", [(0, True), (3, False)])
def test_robot_camera_inset_is_skipped_in_robot_camera_view(tmp_path, monkeypatch, view, expect_inset):
    from robot_env.sim import RobotSim
    calls = []
    original = RobotSim.prepare_camera

    def counting(self, *args, **kwargs):
        calls.append(1)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(RobotSim, "prepare_camera", counting)
    run_app(tmp_path, frames=5, view=view)
    assert bool(calls) == expect_inset


def test_parse_script():
    steps = parse_script("W:2;W+A:1;none:0.5;SPACE:0.3")
    assert steps[1].keys == frozenset({pygame.K_w, pygame.K_a}) and steps[2].keys == frozenset()
    assert steps[3].duration == 0.3


@pytest.mark.gui
def test_real_time_factor_is_honest_when_frames_are_slow(tmp_path, monkeypatch):
    """Regression: RTF showed 1.00 at about 9 FPS."""
    import time as _time
    original = App.draw

    def slow_draw(self):
        original(self)
        _time.sleep(0.12)

    monkeypatch.setattr(App, "draw", slow_draw)
    summary, _ = run_app(tmp_path, frames=12, script=parse_script("W:3"))
    assert summary["real_time_factor"] < 0.7


@pytest.mark.parametrize("pose", [(4.78, 0.0, 3.14), (-4.78, 0.0, 0.0), (0.0, 0.48, -1.57),
                                  (1.5, -1.8, 0.0), (-2.5, 1.2, -1.57), (2.5, 3.1, 0.0)])
@pytest.mark.parametrize("view", [0, 2])
def test_viewer_camera_never_ends_up_inside_walls_or_furniture(pose, view):
    """Regression: the chase camera entered interior walls. The camera must have a
    clear line of sight to the robot from wherever it is placed."""
    import math
    import numpy as np
    from robot_env.app import ViewCamera
    from robot_env.sim import RobotSim
    sim = RobotSim()
    x, y, yaw = pose
    sim.reset(x, y, yaw, (0.0, 0.0))
    v = ViewCamera()
    v.mode = view
    for azimuth in (0.0, 90.0, 180.0, 270.0):
        v.azimuth = azimuth
        cam = v.apply(sim)
        az, el = math.radians(cam.azimuth), math.radians(cam.elevation)
        back = np.array([-math.cos(el) * math.cos(az), -math.cos(el) * math.sin(az), -math.sin(el)])
        hit = sim.ray_to_solid(cam.lookat, back)
        assert hit < 0 or hit > cam.distance, (pose, azimuth, hit, cam.distance)
        _assert_near_plane_clear(sim, cam)
    sim.close()


@pytest.mark.gui
def test_episode_clock_stops_at_episode_end(tmp_path):
    from robot_env import config as C
    app = App(1005, screenshot=None, frames=1)
    app.system.advance(C.EPISODE_TIME_LIMIT + 0.5)
    app.update_episode()
    assert app.episode.status == "timeout" and app.episode_time() == C.EPISODE_TIME_LIMIT
    app.system.advance(1.0)
    assert app.episode_time() == C.EPISODE_TIME_LIMIT
    app.run()


class _FakeModelDriver:
    """Stands in for a decision model: always asks for forward motion."""

    name = "fake_model"

    def decide(self, obs):
        from robot_env.types import Command, Decision
        return Decision(Command(0.3, 0.0), obs.seq)


@pytest.mark.gui
def test_window_runs_a_non_manual_driver_without_held_keys(tmp_path):
    """Regression: release-to-stop blocked any non-manual driver."""
    app = App(1006, screenshot=tmp_path / "s.png", frames=60, driver=_FakeModelDriver())
    assert not app.manual_driving
    app.run()
    assert app.system.last_decision.command.v == 0.3
    assert app.system.last_result.reasons == () or "clearance" in app.system.last_result.reasons
    assert app.system.sim.true_velocity()[0] > 0.1  # it actually moves


@pytest.mark.gui
def test_space_still_brakes_a_non_manual_driver(tmp_path):
    app = App(1006, screenshot=tmp_path / "s.png", frames=90, driver=_FakeModelDriver(),
              script=parse_script("none:0.6;SPACE:1.0"))
    app.run()
    assert app.system.last_result.reasons == ("emergency_brake",)
    assert abs(app.system.sim.true_velocity()[0]) < 0.02


@pytest.mark.gui
def test_free_moves_hint_lists_exactly_the_safe_moves(tmp_path):
    """QA finding: when blocked, the panel must say what is still free. Robot facing the
    lab island bench's west end, overlapping it sideways: forward is blocked."""
    from robot_env.safety import is_safe
    from robot_env.types import Command
    app = App(1000, screenshot=None, frames=1)
    app.system.reset(1.79, 2.8, 0.0, (4.0, 4.0))
    app.system.advance(0.1)
    free = app.free_moves()
    v, w = 0.30, 1.0
    expected = [k for k, c in [("W", Command(v, 0)), ("A", Command(0, w)), ("D", Command(0, -w)),
                               ("S", Command(-v * 0.5, 0))] if is_safe(c, app.system.observe())]
    assert free == expected
    assert "W" not in free and "S" in free
    app.run()


def _post(t, k):
    pygame.event.post(pygame.event.Event(t, key=k, mod=0, unicode="", scancode=0))


@pytest.mark.gui
@pytest.mark.parametrize("reset_key", [pygame.K_r, pygame.K_n])
def test_reset_keeps_a_held_space_brake(tmp_path, reset_key):
    """Regression: Space down, R (or N) down, W down, no Space up -> must stay braked."""
    app = App(1006, screenshot=None, frames=40)
    _post(pygame.KEYDOWN, pygame.K_SPACE)
    _post(pygame.KEYDOWN, reset_key)
    _post(pygame.KEYUP, reset_key)
    _post(pygame.KEYDOWN, pygame.K_w)
    app.run()
    assert app.input.emergency_brake and app.input.space_held
    assert app.system.last_result.reasons == ("emergency_brake",)
    assert abs(app.system.sim.true_velocity()[0]) < 0.02


@pytest.mark.gui
def test_scripted_keys_move_then_release_stops(tmp_path):
    app = App(1005, screenshot=None, frames=300, script=parse_script("D:1.5;W:1.5;none:1.2"))  # about 4.2 s of script
    x0, y0, _ = app.system.sim.true_pose()
    app.run()
    x1, y1, _ = app.system.sim.true_pose()
    assert ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5 > 0.2  # it really moved
    assert abs(app.system.sim.true_velocity()[0]) < 0.02  # and released keys stopped it
    assert app.system.collisions == 0


@pytest.mark.gui
@pytest.mark.parametrize("clear", ["r", "n", "focus"])
def test_brake_holds_when_a_forgotten_key_is_still_down(tmp_path, clear):
    """Regression: W down, Space down/up, R (or N, or focus loss and gain),
    D down with no W up -> must stay braked."""
    app = App(1006, screenshot=None, frames=45)
    _post(pygame.KEYDOWN, pygame.K_w)
    _post(pygame.KEYDOWN, pygame.K_SPACE)
    _post(pygame.KEYUP, pygame.K_SPACE)
    if clear == "focus":
        pygame.event.post(pygame.event.Event(pygame.WINDOWFOCUSLOST))
        pygame.event.post(pygame.event.Event(pygame.WINDOWFOCUSGAINED))
    else:
        key = pygame.K_r if clear == "r" else pygame.K_n
        _post(pygame.KEYDOWN, key)
        _post(pygame.KEYUP, key)
    _post(pygame.KEYDOWN, pygame.K_d)
    app.run()
    assert app.input.emergency_brake
    assert app.system.last_result.reasons == ("emergency_brake",)
    assert abs(app.system.sim.true_velocity()[1]) < 0.05


@pytest.mark.gui
def test_hint_when_a_held_key_was_forgotten_by_a_restart(tmp_path):
    app = App(1006, screenshot=None, frames=30)
    _post(pygame.KEYDOWN, pygame.K_w)
    _post(pygame.KEYDOWN, pygame.K_r)
    _post(pygame.KEYUP, pygame.K_r)
    app.run()
    assert pygame.K_w in app.input.pending_release and not app.input.drive_input_held
    assert abs(app.system.sim.true_velocity()[0]) < 0.02  # the forgotten key does not drive


def _camera_position(cam):
    import math
    import numpy as np
    az, el = math.radians(cam.azimuth), math.radians(cam.elevation)
    back = np.array([-math.cos(el) * math.cos(az), -math.cos(el) * math.sin(az), -math.sin(el)])
    return np.array(cam.lookat) + back * cam.distance, back


def _assert_near_plane_clear(sim, cam):
    """Conservative check beyond the single centre ray: the four near-plane corners and a
    small neighbourhood around the camera (right, left, up, down, behind) must be clear, so
    no edge of the image clips into a wall the centre ray just misses."""
    import math
    import numpy as np
    pos, back = _camera_position(cam)
    fwd = -back
    right = np.cross(fwd, (0.0, 0.0, 1.0))
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    m = sim.model
    near = m.vis.map.znear * m.stat.extent
    half_h = near * math.tan(math.radians(m.vis.global_.fovy) / 2)
    half_w = half_h * 1280 / 720
    probes = [fwd * near + right * sx * half_w + up * sy * half_h for sx in (-1, 1) for sy in (-1, 1)]
    probes += [d * 0.03 for d in (right, -right, up, -up, back)]  # 3 cm bubble around the camera
    for p in probes:
        length = float(np.linalg.norm(p))
        hit = sim.ray_to_solid(pos, p / length)
        assert hit < 0 or hit > length, (pos, p, hit)


@pytest.mark.parametrize("pose,command", [
    ((-4.0, 0.0, 0.0), (0.3, 0.0)),        # straight down the corridor
    ((-2.5, 0.2, 1.5708), (0.3, 0.0)),     # through the office doorway
    ((4.6, 0.0, 3.1416), (0.0, 1.0)),      # turning in place at the corridor's end wall
    ((-2.0, -3.6, 1.5708), (0.3, 0.6)),    # arcing in the storage aisle between two shelves
])
def test_chase_camera_is_clear_every_frame_and_never_pops_outward(pose, command):
    """Smoothing applies to the desired view, but clearance is a hard limit
    every frame; outward motion must be gradual (no pops)."""
    from robot_env.app import ViewCamera
    from robot_env.system import RobotSystem
    s = RobotSystem()
    s.reset(*pose, (4.0, -4.0))
    v = ViewCamera()
    prev_dist = None
    for _ in range(150):  # 2.5 s at 60 FPS
        s.drive(*command)
        s.advance(1 / 60)
        cam = v.apply(s.sim, 1 / 60)
        _, back = _camera_position(cam)
        hit = s.sim.ray_to_solid(cam.lookat, back)
        assert hit < 0 or hit >= cam.distance, (hit, cam.distance)
        _assert_near_plane_clear(s.sim, cam)
        if prev_dist is not None:
            assert cam.distance - prev_dist <= 0.05, (prev_dist, cam.distance)  # no outward pop
        prev_dist = cam.distance
    s.close()


def test_chase_camera_heading_filter_wraps_the_short_way():
    import math
    from robot_env.app import ViewCamera
    from robot_env.sim import RobotSim
    sim = RobotSim(include_obstacles=False)
    v = ViewCamera()
    sim.reset(0.0, 0.0, math.radians(179.0), (4.0, 4.0))
    v.apply(sim, 1 / 60)
    sim.reset(0.0, 0.0, math.radians(-179.0), (4.0, 4.0))  # 2 degrees further, across the seam
    cam = v.apply(sim, 1 / 60)
    assert abs(((cam.azimuth - 179.0) + 180.0) % 360.0 - 180.0) < 3.0  # moved about 2 degrees, not 358
    sim.close()


def test_chase_camera_smooths_small_heading_wobble_and_snaps_on_reset():
    import math
    from robot_env.app import ViewCamera
    from robot_env.sim import RobotSim
    sim = RobotSim(include_obstacles=False)
    v = ViewCamera()
    sim.reset(0.0, 0.0, 0.0, (4.0, 4.0))
    v.apply(sim, 1 / 60)
    sim.reset(0.0, 0.0, math.radians(3.0), (4.0, 4.0))  # a sudden 3 degree wobble
    cam = v.apply(sim, 1 / 60)
    assert abs(cam.azimuth) < 0.5  # one frame later the view has barely moved
    v.reset()
    cam = v.apply(sim, 1 / 60)
    assert abs(cam.azimuth - 3.0) < 1e-6  # after a reset it snaps to the true heading
    sim.close()


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_chase_camera_filters_do_not_depend_on_frame_rate(seed):
    """The view follows a 30 degree turn the same way whether frames are even (60 FPS) or
    uneven (anything from 144 FPS to 20 FPS): the filters use the real frame time."""
    import math
    import numpy as np
    from robot_env.app import ViewCamera
    from robot_env.sim import RobotSim
    sim = RobotSim(include_obstacles=False)

    def follow(frame_times):
        v = ViewCamera()
        sim.reset(0.0, 0.0, 0.0, (4.0, 4.0))
        v.apply(sim, 1 / 60)
        sim.reset(0.0, 0.0, math.radians(30.0), (4.0, 4.0))
        for dt in frame_times:
            cam = v.apply(sim, dt)
        return cam.azimuth

    total = 0.3
    even = follow([1 / 60] * round(total * 60))
    rng = np.random.default_rng(seed)
    uneven = list(rng.uniform(1 / 144, 1 / 20, 40))
    uneven = [dt for dt in np.cumsum(uneven) if dt < total]
    uneven = np.diff([0.0] + uneven + [total])
    assert abs(follow(list(uneven)) - even) < 0.01  # degrees
    assert 30.0 * (1 - math.exp(-total / ViewCamera.YAW_TAU)) == pytest.approx(even, abs=0.01)
    sim.close()


# ----- speed levels in the window -----

def _key_event(app, t, k):
    app.input.handle_event(pygame.event.Event(t, key=k, mod=0, unicode="", scancode=0))


def _open_app(level=None):
    """An app in open space (the robot faces down the empty corridor), driven frame by frame."""
    from robot_env import config as C
    app = App(1000, screenshot=None, frames=1,
              speed_level=C.DEFAULT_SPEED_LEVEL if level is None else level)
    app.system.reset(-4.3, 0.0, 0.0, (4.5, 0.0))
    return app


@pytest.mark.gui
def test_shift_and_level_keys_retarget_a_held_drive_key_without_a_new_press():
    """Pressing Shift or +/- while W is held changes the requested command at once and the
    approved target by the next control tick; the applied command stays rate limited."""
    from robot_env import config as C
    app = _open_app(0)
    s = app.system
    _key_event(app, pygame.KEYDOWN, pygame.K_w)
    for _ in range(60):
        app.simulate(1 / 60)
    assert s.last_result.command.v == pytest.approx(C.SPEED_LEVELS[0][0])
    for k, expect in [(pygame.K_EQUALS, C.SPEED_LEVELS[1][0]), (pygame.K_LSHIFT, C.SPEED_LEVELS[-1][0])]:
        _key_event(app, pygame.KEYDOWN, k)
        t0, before = s.time, s.applied.v
        app.simulate(C.CONTROL_PERIOD)  # one control period always contains a control tick
        assert s.requested.v == pytest.approx(expect)  # sent at once, no new W press
        assert s.last_result.command.v == pytest.approx(expect)  # approved by the next tick
        assert s.applied.v - before <= C.SMOOTH_ACCEL * (s.time - t0) + 1e-9  # no spike
    _key_event(app, pygame.KEYUP, pygame.K_LSHIFT)
    app.simulate(1 / 60)
    assert s.requested.v == pytest.approx(C.SPEED_LEVELS[1][0])  # back to the selected level
    app.run()


@pytest.mark.gui
def test_level_keys_never_release_the_brake_in_the_window():
    app = _open_app()
    _key_event(app, pygame.KEYDOWN, pygame.K_SPACE)
    _key_event(app, pygame.KEYUP, pygame.K_SPACE)
    for k in (pygame.K_EQUALS, pygame.K_MINUS, pygame.K_KP_PLUS, pygame.K_LSHIFT):
        _key_event(app, pygame.KEYDOWN, k)
        _key_event(app, pygame.KEYUP, k)
    for _ in range(30):
        app.simulate(1 / 60)
    assert app.input.emergency_brake and app.system.last_result.reasons == ("emergency_brake",)
    app.run()


@pytest.mark.gui
def test_restart_and_next_goal_keep_the_selected_level():
    app = _open_app()
    _key_event(app, pygame.KEYDOWN, pygame.K_MINUS)
    level = app.input.level
    app.new_episode(app.seed)
    app.new_episode(app.seed + 1)
    assert app.input.level == level
    app.run()


@pytest.mark.gui
def test_status_line_shows_selected_and_effective_levels():
    from robot_env import config as C
    app = _open_app()
    n = len(C.SPEED_LEVELS)
    assert app.speed_level_text().startswith(f"speed level {C.DEFAULT_SPEED_LEVEL + 1}/{n}: ")
    _key_event(app, pygame.KEYDOWN, pygame.K_RSHIFT)
    assert f"Shift: level {n}" in app.speed_level_text()
    app.run()


@pytest.mark.gui
def test_free_moves_use_the_level_in_effect():
    """At the slowest level the hint must test the slow commands, not the default ones."""
    from robot_env import config as C
    from robot_env.safety import is_safe
    from robot_env.types import Command
    app = App(1000, screenshot=None, frames=1, speed_level=0)
    app.system.reset(1.79, 2.8, 0.0, (4.0, 4.0))
    app.system.advance(0.1)
    v, w = C.SPEED_LEVELS[0]
    expected = [k for k, c in [("W", Command(v, 0)), ("A", Command(0, w)), ("D", Command(0, -w)),
                               ("S", Command(-v * C.MANUAL_REVERSE_FACTOR, 0))] if is_safe(c, app.system.observe())]
    assert app.free_moves() == expected
    app.run()


def test_speed_level_command_line_option():
    from robot_env import config as C
    from robot_env.app import main
    summary = main(["--frames", "2", "--speed-level", "1"])
    assert summary["status"] == "running"
    with pytest.raises(SystemExit):
        main(["--frames", "2", "--speed-level", str(len(C.SPEED_LEVELS) + 1)])


@pytest.mark.gui
def test_camera_stays_continuous_while_changing_level_and_shift():
    """Changing the speed level and pressing or releasing Shift while driving never makes the
    chase camera pop; the status line updates in the same frame."""
    import math
    from robot_env import config as C
    app = _open_app(0)
    app.view.mode = 0
    cams = []

    def frame():
        app.simulate(1 / 60)
        cam = app.view.apply(app.system.sim, 1 / 60)
        pos, _ = _camera_position(cam)
        cams.append((pos.copy(), cam.azimuth, cam.elevation))

    _key_event(app, pygame.KEYDOWN, pygame.K_w)
    for _ in range(40):
        frame()
    events = [(pygame.KEYDOWN, pygame.K_EQUALS), (pygame.KEYDOWN, pygame.K_LSHIFT), (pygame.KEYUP, pygame.K_LSHIFT),
              (pygame.KEYDOWN, pygame.K_MINUS)]
    for t, k in events:
        _key_event(app, t, k)
        text = app.speed_level_text()
        assert text.startswith(f"speed level {app.input.level + 1}/{len(C.SPEED_LEVELS)}")
        assert ("Shift: level" in text) == app.input.boosted
        for _ in range(40):
            frame()
    settled = cams[39:]  # from the frame before the first key change (the start-up framing settles first)
    for (p0, a0, e0), (p1, a1, e1) in zip(settled, settled[1:]):
        assert math.dist(p0, p1) <= 0.06  # at most about one frame of robot travel plus smoothing
        assert abs(((a1 - a0) + 180) % 360 - 180) <= 1.0 and abs(e1 - e0) <= 1.0
    app.run()


# ----- chase camera line of sight near furniture -----

CAMERA_POSES = {  # robot backed up to furniture or a wall: the chase camera's usual spot is taken
    "chair": (-3.8, 3.2, -1.5708),
    "desk": (-3.3, 3.85, -1.5708),
    "shelf": (-4.25, -3.0, 0.0),
    "doorway": (-2.5, 0.75, -1.5708),
    "wall": (4.6, 0.0, 3.1416),
}


def _sees_robot_and_ahead(sim, view, cam) -> bool:
    import numpy as np
    x, y, yaw = sim.true_pose()
    pos, _ = _camera_position(cam)
    for p in view._sight_targets(sim, x, y, yaw):
        d = p - pos
        n = float(np.linalg.norm(d))
        hit = sim.ray_to_view_blocker(pos, d / n)
        if 0 <= hit < n - 0.03:
            return False
    return True


@pytest.mark.parametrize("name", sorted(CAMERA_POSES))
def test_chase_camera_keeps_the_robot_and_the_way_ahead_in_view(name):
    """Regression (hands-on review): next to a desk and chair the chase camera sat between the
    chair's arms and hid the robot. At every frame: the camera is outside all geometry
    (including visual detail), and the robot's corners and the way ahead are visible, or the
    camera reports it is blocked. The chosen detour never flips side."""
    from robot_env.app import ViewCamera
    from robot_env.layout import RoomMap
    from robot_env.sim import RobotSim
    sim = RobotSim()
    x, y, yaw = CAMERA_POSES[name]
    assert RoomMap(sim.model).point_clearance(x, y) > 0.2126  # a pose the robot can really be in
    sim.reset(x, y, yaw, (0.0, 0.0))
    view = ViewCamera()
    sides = []
    for frame in range(120):
        cam = view.apply(sim, 1 / 60)
        pos, back = _camera_position(cam)
        hit = sim.ray_to_view_blocker(cam.lookat, back)
        assert hit < 0 or hit >= cam.distance, (name, frame)  # never inside anything
        assert view.blocked or _sees_robot_and_ahead(sim, view, cam), (name, frame)
        sides.append(view._detour[0])
    flips = sum(1 for a, b in zip(sides, sides[1:]) if a * b < 0)
    assert flips == 0, sides
    assert not view.blocked
    sim.close()


def test_chase_camera_returns_to_the_normal_view_after_leaving_furniture():
    """Leaving the chair: the camera eases back to the normal chase angle and distance."""
    from robot_env.app import ViewCamera
    from robot_env.sim import RobotSim
    sim = RobotSim()
    view = ViewCamera()
    x, y, yaw = CAMERA_POSES["chair"]
    sim.reset(x, y, yaw, (0.0, 0.0))
    for _ in range(60):
        view.apply(sim, 1 / 60)
    assert view.cam.elevation < -40 or view._detour != (0.0, None)  # detouring around the chair
    for i in range(1, 61):  # slide 1.2 m away from the chair, into open office floor
        sim.reset(x, y - 0.02 * i, yaw, (0.0, 0.0))
        view.apply(sim, 1 / 60)
    for _ in range(180):
        cam = view.apply(sim, 1 / 60)
    # back to the user's angle and as far out as the room allows (the nearest wall may cap it)
    assert abs(cam.elevation - view.elevation) < 2.0
    room = min(view.distance, view._clear_distance(sim, cam.azimuth, cam.elevation),
               view._beam_room(sim, cam.azimuth, cam.elevation))  # side clearance counts too
    assert cam.distance == pytest.approx(room, abs=0.05)
    assert view._detour == (0.0, None)
    sim.close()


# ----- step 2: steady camera, zoom, window -----
@pytest.mark.parametrize("seed,view", [(1003, "chase"), (1007, "orbit")])
def test_camera_moves_smoothly_on_real_routes(seed, view):
    """The routes that made the camera jump up to 2.9 m in one frame: no frame moves the camera
    more than 0.10 m, the camera stays a sensible distance away (never closer than 0.4 m, the
    robot always visible), and a still robot gives a still camera."""
    import importlib
    jitter = importlib.import_module("tools.camera_jitter")
    r = jitter.measure(seed, view, seconds=25.0)
    assert r["max_jump_m"] <= 0.10, r
    assert r["mean_distance_m"] >= 1.0, r
    assert r["min_distance_m"] >= 0.40 - 1e-3 and r["robot_hidden_frames"] == 0, r  # never a useless close-up
    still = jitter.measure(seed, view, still=2.0)
    assert still["settled_last_s_max_m"] <= 0.002, still


def test_wheel_always_changes_the_requested_zoom_and_tight_spaces_show_a_cue():
    from robot_env.app import ViewCamera
    from robot_env.sim import RobotSim
    sim = RobotSim()
    view = ViewCamera()
    x, y, yaw = CAMERA_POSES["shelf"]
    sim.reset(x, y, yaw, (0.0, 0.0))
    for _ in range(60):
        view.apply(sim, 1 / 60)
    shown = view.cam.distance
    for _ in range(5):  # zoom out where walls already limit the view
        view.handle_event(pygame.event.Event(pygame.MOUSEWHEEL, x=0, y=-1, flipped=False))
    assert view.distance > 1.6  # the request changed even though the view cannot follow
    for _ in range(60):
        view.apply(sim, 1 / 60)
    assert view.limited and view.cam.distance <= shown + 0.05
    for _ in range(30):  # zoom in: the view follows as far as the room allows
        view.handle_event(pygame.event.Event(pygame.MOUSEWHEEL, x=0, y=1, flipped=False))
    for _ in range(60):
        view.apply(sim, 1 / 60)
    assert view.cam.distance == pytest.approx(min(view.distance, view.room), abs=0.06)
    assert view.cam.distance < shown - 0.3
    sim.close()


def _zoom_response(fps, notches, start=1.6):
    """The displayed distance per frame after `notches` wheel notches in open space (the office)."""
    from robot_env.app import ViewCamera
    from robot_env.sim import RobotSim
    sim = RobotSim()
    view = ViewCamera()
    view.distance = start
    sim.reset(-1.8, 2.6, 0.0, (4.0, 4.0))  # 2.5 m of room behind the robot
    for _ in range(fps):  # settle
        view.apply(sim, 1 / fps)
    for _ in range(abs(notches)):
        view.handle_event(pygame.event.Event(pygame.MOUSEWHEEL, x=0, y=1 if notches > 0 else -1, flipped=False))
    shown = []
    for _ in range(fps):  # one second
        view.apply(sim, 1 / fps)
        shown.append(view.cam.distance)
    sim.close()
    return start, view.distance, shown


@pytest.mark.parametrize("fps", [30, 60, 120])
@pytest.mark.parametrize("notches", [1, -1])
def test_zoom_follows_the_wheel_within_a_tenth_of_a_second(fps, notches):
    """A notch reaches 90 percent within 120 ms and 95 percent within 160 ms at any frame rate,
    with no overshoot (at most 1 percent of the step) and no drift once settled."""
    start, target, shown = _zoom_response(fps, notches)
    step = target - start
    progress = [(d - start) / step for d in shown]
    t90 = next(k for k, f in enumerate(progress) if f >= 0.9) / fps + 1 / fps
    t95 = next(k for k, f in enumerate(progress) if f >= 0.95) / fps + 1 / fps
    assert t90 <= 0.12 + 1e-9 and t95 <= 0.16 + 1e-9, (t90, t95)
    assert max(progress) <= 1.01
    assert max(shown[-fps // 4:]) - min(shown[-fps // 4:]) < 0.001  # still at rest


def test_rapid_notches_go_straight_to_the_final_zoom():
    """Ten quick notches aim at the final distance at once (no queue of ten animations) and the
    view never turns back on the way."""
    start, target, shown = _zoom_response(60, 10, start=2.5)  # inside the room (2.95 m): no rebase
    assert target == pytest.approx(2.5 * 0.88 ** 10)
    steps = np.diff([start] + shown)
    assert (steps <= 1e-9).all()  # zooming in all the way, never back out
    assert shown[-1] == pytest.approx(target, abs=0.01)
    assert sum(1 for d in shown if abs(d - target) > 0.05 * abs(start - target)) / 60 <= 0.16


def test_camera_recovers_its_distance_within_a_second_without_overshoot():
    """Backed up to the corridor's end wall the camera is held close; after the robot drives
    1.2 m away (1 s), the camera is back at the distance the room allows within 1 s, never beyond it."""
    from robot_env.app import ViewCamera
    from robot_env.sim import RobotSim
    sim = RobotSim()
    view = ViewCamera()
    view.distance = 3.0
    x, y, yaw = CAMERA_POSES["wall"]  # facing away from the wall: the camera's spot is in it
    sim.reset(x, y, yaw, (0.0, 0.0))
    for _ in range(60):
        view.apply(sim, 1 / 60)
    held = view.cam.distance
    for i in range(1, 61):
        sim.reset(x - 0.02 * i, y, yaw, (0.0, 0.0))
        view.apply(sim, 1 / 60)
    dists = []
    for _ in range(120):
        dists.append(view.apply(sim, 1 / 60).distance)
    final = min(view.distance, view.room)
    assert final > held + 0.2  # there really was something to recover
    assert dists[59] >= final - 0.05  # back within 1 s of leaving
    assert max(dists) <= final + 0.02 and max(dists) <= view.distance + 1e-9  # no overshoot
    assert all(b >= a - 1e-6 for a, b in zip(dists, dists[1:]))  # a steady glide out, no wobble
    sim.close()


# ----- third-person camera: the user's angle and zoom are the authority (spring-arm rules) -----
OPEN_OFFICE = (-1.8, 2.6, 0.0)  # 2.95 m of room behind the robot
SEED_1003_START = (-1.31, 0.17, 0.576)  # the corridor near the office door: walls hold the camera in


def _wheel(view, y, times=1):
    for _ in range(times):
        view.handle_event(pygame.event.Event(pygame.MOUSEWHEEL, x=0, y=y, flipped=False))


def _drag(view, dx, dy):
    view.handle_event(pygame.event.Event(pygame.MOUSEMOTION, rel=(dx, dy), buttons=(1, 0, 0), pos=(640, 360)))


def _settled_view(pose, mode=0, frames=60, goal=(4.0, 4.0)):
    from robot_env.app import ViewCamera
    from robot_env.sim import RobotSim
    sim = RobotSim()
    sim.reset(*pose, goal)
    view = ViewCamera()
    view.mode = mode
    for _ in range(frames):
        view.apply(sim, 1 / 60)
    return sim, view


@pytest.mark.parametrize("name", ["shelf", "wall", "chair"])
def test_first_wheel_notch_in_moves_a_camera_the_walls_hold(name):
    """Regression (owner: zoom not working): after scrolling out against walls, 7 to 13 notches in
    did nothing. Zooming in now starts from what is shown, so the first notch moves the camera."""
    sim, view = _settled_view(CAMERA_POSES[name])
    _wheel(view, -1, 5)
    for _ in range(60):
        view.apply(sim, 1 / 60)
    before = view.cam.distance
    assert view.distance > before + 0.3  # the request is far beyond what the walls allow
    _wheel(view, 1)
    for _ in range(20):
        view.apply(sim, 1 / 60)
    assert view.cam.distance < before * 0.93, (before, view.cam.distance)
    sim.close()


def test_zoom_in_magnifies_the_robot():
    """Regression: the lens widened as the user zoomed in, so five notches grew the robot by 17
    percent instead of 90. A distance the user chose keeps the normal lens."""
    import math
    sim, view = _settled_view(OPEN_OFFICE)
    size0 = 1 / (view.cam.distance * math.tan(math.radians(view.fovy) / 2))
    _wheel(view, 1, 5)
    for _ in range(60):
        view.apply(sim, 1 / 60)
    size1 = 1 / (view.cam.distance * math.tan(math.radians(view.fovy) / 2))
    assert size1 / size0 >= 1.8 and view.fovy == pytest.approx(45.0, abs=0.1)
    sim.close()


def test_lens_widens_only_when_walls_hold_the_camera_close():
    sim, view = _settled_view(CAMERA_POSES["wall"], frames=90)  # backed up to the corridor's end wall
    assert view.cam.distance < 0.8 and view.fovy > 55.0
    sim.close()


@pytest.mark.parametrize("pose", [OPEN_OFFICE, SEED_1003_START])
@pytest.mark.parametrize("mode", [0, 2])
def test_zoom_out_never_pulls_the_camera_in_or_turns_it(pose, mode):
    """Regression: a larger zoom request made the line-of-sight check fail at a distance the camera
    never showed, so zooming out swung, tilted, and pulled the camera in (seed 1003 start:
    1.11 m -> 0.92 m and a 20 degree swing)."""
    sim, view = _settled_view(pose, mode)
    d0, az0, el0 = view.cam.distance, view.cam.azimuth, view.cam.elevation
    for _ in range(6):
        _wheel(view, -1)
        for _ in range(20):
            cam = view.apply(sim, 1 / 60)
            assert cam.distance >= d0 - 0.01, (d0, cam.distance)
            assert abs(((cam.azimuth - az0) + 180) % 360 - 180) <= 1.0 and abs(cam.elevation - el0) <= 1.0
    sim.close()


@pytest.mark.parametrize("mode", [0, 2])
def test_drag_turns_and_tilts_the_view_the_same_frame(mode):
    """Regression: the camera's speed cap also slowed the user's drag (a 180 degree flick lagged
    146 degrees at release and arrived 2.3 s later; tilting followed at 40 degrees/s)."""
    sim, view = _settled_view(OPEN_OFFICE, mode)
    for _ in range(15):  # a fast flick: 40 px per frame, 12 degrees
        az0 = view.cam.azimuth
        _drag(view, 40, 0)
        cam = view.apply(sim, 1 / 60)
        assert ((cam.azimuth - az0) + 180) % 360 - 180 == pytest.approx(-12.0, abs=0.01)
    for _ in range(20):  # tilt up to the shallowest pitch
        el0 = view.cam.elevation
        _drag(view, 0, -10)
        cam = view.apply(sim, 1 / 60)
        assert cam.elevation == pytest.approx(min(el0 + 3.0, -3.0), abs=0.01)
    cam = view.apply(sim, 1 / 60)
    assert cam.elevation == pytest.approx(-3.0) and view._detour == (0.0, None)
    sim.close()


@pytest.mark.parametrize("pose,dy", [((0.0, 0.0, 1.5708), -10), (CAMERA_POSES["wall"], 10)])
def test_the_automatic_tilt_never_steps_against_a_slow_drag(pose, dy):
    """Regression (review): a mouse that reports less often than frames are drawn left frames of a
    drag without a motion event, and on those the automatic tilt stepped back against the hand."""
    sim, view = _settled_view(pose)
    view.handle_event(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=(640, 360)))
    els = []
    for frame in range(60):
        if frame % 3 == 0:
            _drag(view, 0, dy)
        els.append(view.apply(sim, 1 / 60).elevation)
    steps = np.diff(els)
    assert (steps * (-dy) >= -1e-9).all(), steps  # never against the drag (up for dy < 0)
    sim.close()


def test_a_tilt_up_with_room_for_min_view_is_kept_after_the_release():
    """Regression (review): the automatic tilt-back margin (TILT_BACK) also judged the user's own
    tilt, so in orbit next to the chair (about 0.7 to 1.1 m of room at every pitch) a tilt up to -3
    sank back to -22."""
    sim, view = _settled_view(CAMERA_POSES["chair"], mode=2)
    view.handle_event(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=(640, 360)))
    for _ in range(25):
        _drag(view, 0, -10)
        view.apply(sim, 1 / 60)
    view.handle_event(pygame.event.Event(pygame.MOUSEBUTTONUP, button=1, pos=(640, 360)))
    for _ in range(90):
        cam = view.apply(sim, 1 / 60)
    assert view.elevation == -3.0 and cam.elevation == pytest.approx(-3.0, abs=1.0)
    sim.close()


@pytest.mark.parametrize("name", ["wall", "chair", "shelf"])
def test_tilting_down_a_camera_the_walls_hold_steep_follows_the_user(name):
    """Backed up to a wall the camera looks down steeper than asked; tilting down further leaves it
    there until the user's pitch is the steeper one, then follows it, never past PITCH[0]."""
    sim, view = _settled_view(CAMERA_POSES[name])
    lo, hi = view.PITCH
    steep = view.cam.elevation
    assert steep < view.elevation - 10  # the walls really hold it steeper
    for _ in range(25):
        _drag(view, 0, 10)
        cam = view.apply(sim, 1 / 60)
        assert lo - 1e-9 <= cam.elevation <= hi and cam.elevation <= min(steep, view.elevation) + 1e-6
    assert view.elevation == lo and cam.elevation == pytest.approx(lo)
    sim.close()


@pytest.mark.parametrize("offset", [(0.0, 0.0), (-0.1, 0.0), (-0.2, 0.0)])
@pytest.mark.parametrize("mode", [0, 2])
def test_the_goal_marker_never_blocks_the_view(offset, mode):
    """Regression: with the goal's thin pole under or just behind the robot the camera collapsed to
    2 cm, looking straight down (blocked)."""
    x, y, yaw = OPEN_OFFICE
    sim, view = _settled_view(OPEN_OFFICE, mode, goal=(x + offset[0], y + offset[1]))
    assert view.cam.distance == pytest.approx(view.distance, abs=0.02) and not view.blocked
    assert view.cam.elevation == pytest.approx(view.elevation) and view._detour == (0.0, None)
    sim.close()


def test_a_drag_that_starts_on_a_panel_does_not_turn_the_view():
    from robot_env.app import ViewCamera
    view = ViewCamera()
    view.ignore_drag = True  # the window gave the press to the Options menu
    view.handle_event(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=(50, 90)))
    _drag(view, 40, 20)
    assert (view.azimuth, view.elevation) == (view.AZIMUTH, view.ELEVATION)
    view.handle_event(pygame.event.Event(pygame.MOUSEBUTTONUP, button=1, pos=(50, 90)))
    _drag(view, 40, 0)
    assert view.azimuth == pytest.approx(view.AZIMUTH - 12.0)
    view.ignore_drag = True  # a menu press, then the window loses focus before the release
    view.handle_event(pygame.event.Event(pygame.WINDOWFOCUSLOST))
    _drag(view, 40, 0)
    assert view.azimuth == pytest.approx(view.AZIMUTH - 24.0)


@pytest.mark.parametrize("mode", [1, 3])
def test_the_mouse_never_changes_the_third_person_view_from_another_view(mode):
    """Nudging the mouse in the top or robot-camera view must not leave the chase view turned."""
    from robot_env.app import ViewCamera
    view = ViewCamera()
    view.mode = mode
    _drag(view, 100, -50)
    _wheel(view, -1)
    assert (view.azimuth, view.elevation, view.distance) == (view.AZIMUTH, view.ELEVATION, view.DISTANCE)


@pytest.mark.gui
def test_restart_restores_the_default_view_and_clicks_use_drawing_coordinates():
    app = App(1000, screenshot=None, frames=1)
    app.view.azimuth, app.view.elevation, app.view.distance = 90.0, -60.0, 3.0
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_r, mod=0, unicode="r", scancode=0))
    app.handle_events()
    assert (app.view.azimuth, app.view.elevation, app.view.distance) == (0.0, -22.0, 1.6)
    w, h = app.display.window_size
    sw, sh = app.screen.get_size()
    assert app._surface_pos((w, h)) == (sw, sh) and app._surface_pos((w // 2, h // 2)) == (sw * (w // 2) // w, sh * (h // 2) // h)
    app.run()


@pytest.mark.gui
def test_window_resizes_and_renders_at_the_capped_size():
    from robot_env.display import MAX_RENDER_PIXELS, render_size
    app = App(1000, screenshot=None, frames=3)
    for size in ((1600, 900), (3440, 1440), (960, 540)):
        app.display.window.size = size
        pygame.event.post(pygame.event.Event(pygame.WINDOWSIZECHANGED, x=size[0], y=size[1]))
        app.handle_events()
        w, h = app.screen.get_size()
        assert (w, h) == render_size(app.display.window_size)
        assert w * h <= MAX_RENDER_PIXELS * 1.01 and abs(w / h - size[0] / size[1]) < 0.02
        assert (app.renderer.width, app.renderer.height) == (w, h)
        app.draw()  # panels lay out inside the new size
        app.display.present()
    app.run()


@pytest.mark.gui
def test_h_cycles_compact_full_and_no_panels():
    app = App(1000, screenshot=None, frames=1)
    seen = [app.hud]
    for _ in range(3):
        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_h, mod=0, unicode="h", scancode=0))
        app.handle_events()
        app.draw()
        seen.append(app.hud)
    assert seen == ["compact", "full", "none", "compact"]
    app.run()


@pytest.mark.gui
def test_window_presents_through_an_accelerated_renderer():
    """Smoke test only: the window uses an accelerated renderer and presents frames. The
    frame-time gate is tools/fps_protocol.py on an idle, focused desktop (timing here depends on
    whatever else the desktop is doing)."""
    app = App(1000, screenshot=None, frames=30, script=parse_script("W:1"))
    app.run()
    assert app.display.driver in ("direct3d11", "direct3d12", "opengl", "metal")
    assert len(app.frame_times) == 29 and all(t > 0 for t in app.frame_times)


def test_frame_pacing_follows_a_fixed_schedule():
    """Deterministic pacing with a fake clock: slots are 1/60 s apart, a late frame restarts the
    schedule (no burst of catch-up frames), and the wait sleeps coarsely then spins 2 ms."""
    from robot_env.app import FPS, _wait_until, present_deadline
    period = 1 / FPS
    assert present_deadline(None, 5.0) == 5.0  # first frame: now
    assert present_deadline(5.0, 5.004) == pytest.approx(5.0 + period)  # on time: next slot
    assert present_deadline(5.0, 5.0 + 2.5 * period) == pytest.approx(5.0 + 2.5 * period)  # late: resync
    now = [10.0]
    sleeps = []

    def clock():
        now[0] += 0.0001  # each look at the clock costs a little time
        return now[0]

    def sleep(t):
        sleeps.append(t)
        now[0] += t
    _wait_until(10.0 + period, clock, sleep)
    assert 10.0 + period <= now[0] <= 10.0 + period + 0.0002
    assert len(sleeps) == 1 and sleeps[0] == pytest.approx(period - 0.002, abs=1e-3)
    calls = []
    _wait_until(5.0, lambda: (calls.append(1), 6.0)[1], sleep)  # already late: no wait at all
    assert len(calls) == 1


def test_minimum_window_size_is_enforced_and_f11_round_trips():
    """With a fake SDL window: a resize below 960x540 is pushed back to the minimum, and F11
    switches to desktop fullscreen and back."""
    from robot_env.display import MIN_SIZE, Display, render_size

    class FakeWindow:
        def __init__(self):
            self.size = (1280, 720)
            self.calls = []

        def set_fullscreen(self, desktop=False):
            self.calls.append(("fullscreen", desktop))
            self.size = (3440, 1440)

        def set_windowed(self):
            self.calls.append(("windowed",))
            self.size = (1280, 720)

    d = Display.__new__(Display)
    d.window, d.fullscreen = FakeWindow(), False
    d.surface = pygame.Surface(render_size((1280, 720)))
    d.window.size = (700, 400)
    assert d.handle_event(pygame.event.Event(pygame.WINDOWSIZECHANGED, x=700, y=400)) is True
    assert d.window.size == MIN_SIZE and d.surface.get_size() == render_size(MIN_SIZE)
    f11 = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_F11, mod=0, unicode="", scancode=0)
    assert d.handle_event(f11) is True and d.fullscreen and d.window.calls[-1] == ("fullscreen", True)
    assert d.surface.get_size() == render_size((3440, 1440))
    assert d.handle_event(f11) is True and not d.fullscreen and d.window.calls[-1] == ("windowed",)
    assert d.surface.get_size() == render_size((1280, 720))


def _app_for_inset(view=0):
    app = App(1000, None, 3, None, view, cats=1, cat_seed=2)
    app._frame_dt = 1 / 60
    return app


@pytest.mark.gui
def test_inset_failure_restores_the_shadow_lights(monkeypatch):
    """The robot-camera inset renders without shadow passes; if that render fails, the lights
    are restored to what the main view chose."""
    app = _app_for_inset()
    sim = app.system.sim
    chosen = {}
    original = app._choose_lights

    def choose(camera):
        original(camera)
        chosen["cast"] = sim.model.light_castshadow.copy()
    monkeypatch.setattr(app, "_choose_lights", choose)

    def broken(*args, **kwargs):
        assert not sim.model.light_castshadow.any()  # no shadow passes for the inset
        raise RuntimeError("inset render failed")
    monkeypatch.setattr(sim, "prepare_camera", broken)
    with pytest.raises(RuntimeError):
        app.draw()
    assert np.array_equal(sim.model.light_castshadow, chosen["cast"])
    close_renderer(app.renderer)
    app.system.close()
    app.display.close()
    pygame.quit()


@pytest.mark.gui
def test_inset_leaves_the_main_view_and_the_robot_camera_unchanged():
    """After an inset render (no shadows, no reflections), the main view and the full robot
    camera render as before it (to the GPU's own frame-to-frame rounding)."""
    import mujoco
    app = _app_for_inset()
    sim = app.system.sim
    cam = mujoco.MjvCamera()
    cam.lookat[:], cam.distance, cam.azimuth, cam.elevation = (-4.0, 4.0, 0.3), 2.5, 135.0, -30.0

    def renders():
        sim.before_render()
        app.renderer.update_scene(sim.data, camera=cam, scene_option=app.overview_option)
        main = app.renderer.render().copy()
        w, h = app._inset.get_size()  # the robot camera at the inset's size (the same renderer)
        return sim.model.light_castshadow.copy(), main, sim.render_camera((h, w)).copy()
    app.draw()  # the main view picks its shadow lights; the inset's scene is prepared...
    app.draw()  # ... and drawn
    cast1, main1, robot1 = renders()
    app._inset_time = -float("inf")
    app.draw()  # the same instant: the same lights, and the inset renders again
    app.draw()
    cast2, main2, robot2 = renders()
    assert np.array_equal(cast1, cast2)
    for before, after in ((main1, main2), (robot1, robot2)):
        # the same render twice can differ by a level in a pixel or two (GPU rounding): at most
        # 2 levels in at most 0.01% of the pixels; a pass left on or off changes far more
        changed = (before != after).any(axis=2)
        assert changed.mean() <= 1e-4 and np.abs(before.astype(int) - after).max() <= 2
    close_renderer(app.renderer)
    app.system.close()
    app.display.close()
    pygame.quit()


@pytest.mark.gui
def test_views_stay_drawn_when_a_renderer_is_replaced():
    """Regression: replacing one renderer (the robot-camera preview changing size with the
    panels, or the main view rebuilt for a new window size) deleted the other renderer's GPU
    objects, and that view went black for good."""
    app = _app_for_inset()
    seen = []
    for hud in ("compact", "full", "compact"):
        app.hud = hud
        app._inset_time = -float("inf")
        app.draw()  # prepares the inset at its new size (a new renderer)...
        app.draw()  # ... and draws it
        seen.append((float(app.renderer.render().mean()), float(app._inset_pixels.mean())))
    app._make_renderer()  # as on a window resize
    app._inset_time = -float("inf")
    app.draw()
    app.draw()
    seen.append((float(app.renderer.render().mean()), float(app._inset_pixels.mean())))
    first = seen[0]
    assert first[0] > 20 and first[1] > 20
    for main, inset in seen[1:]:
        assert main == pytest.approx(first[0], abs=1.0) and inset == pytest.approx(first[1], abs=1.0)
    close_renderer(app.renderer)
    app.system.close()
    app.display.close()
    pygame.quit()


@pytest.mark.gui
@pytest.mark.parametrize("which", ["renderer", "system", "display", "pygame"])
def test_every_cleanup_runs_even_if_one_fails(monkeypatch, which):
    """One cleanup step failing (after a frame failed) still runs every later one, and the frame's
    error is the one raised."""
    import robot_env.app as appmod
    app = App(1000, None, 50, None, 0, cats=1, cat_seed=2)
    ran = []
    real_quit = pygame.quit

    def track(name, real):
        def f(*a, **k):
            ran.append(name)
            if name == which:
                raise OSError(f"{name} close failed")
            return real(*a, **k)
        return f
    monkeypatch.setattr(appmod, "close_renderer", track("renderer", appmod.close_renderer))
    monkeypatch.setattr(app.system, "close", track("system", app.system.close))
    monkeypatch.setattr(app.display, "close", track("display", app.display.close))
    monkeypatch.setattr(appmod.pygame, "quit", track("pygame", pygame.quit))

    def broken():
        raise RuntimeError("draw failed")
    monkeypatch.setattr(app, "draw", broken)
    with pytest.raises(RuntimeError, match="draw failed"):
        app.run()
    assert ran == ["renderer", "system", "display", "pygame"]
    real_quit()


@pytest.mark.gui
def test_a_cleanup_failure_after_a_good_run_is_raised(monkeypatch):
    app = App(1000, None, 3, None, 0, cats=0)
    real = app.display.close

    def failing():
        real()
        raise OSError("display close failed")
    monkeypatch.setattr(app.display, "close", failing)
    with pytest.raises(OSError, match="display close failed"):
        app.run()
    assert not pygame.get_init()


@pytest.mark.gui
@pytest.mark.parametrize("caller_froze", [False, True])
def test_a_failing_frame_still_restores_gc_and_closes_everything(monkeypatch, caller_froze):
    """However the run ends (here a frame that fails to draw), the garbage collector is restored
    (and a freeze the caller made itself is left as it was), and the renderers, the system, the
    window, and pygame are closed."""
    import gc
    app = App(1000, None, 50, None, 0, cats=1, cat_seed=2)
    closed = []
    real_close = app.system.close
    monkeypatch.setattr(app.system, "close", lambda: (closed.append("system"), real_close()))

    def broken():
        raise RuntimeError("draw failed")
    monkeypatch.setattr(app, "draw", broken)
    if caller_froze:
        gc.freeze()
    try:
        with pytest.raises(RuntimeError, match="draw failed"):
            app.run()
        # ours undone; the caller's kept (its count can only fall as frozen objects are freed)
        assert (gc.get_freeze_count() > 0) == caller_froze
    finally:
        if caller_froze:
            gc.unfreeze()
    assert closed == ["system"] and app.renderer._gl_context is None and not pygame.get_init()


def test_fps_protocol_measures_exactly_the_sample_intervals():
    import importlib
    fp = importlib.import_module("tools.fps_protocol")
    times = list(range(fp.WARMUP_FRAMES + fp.SAMPLE_FRAMES))  # what WARMUP + SAMPLE + 1 frames give
    got = fp.sample_intervals(times)
    assert len(got) == fp.SAMPLE_FRAMES and got[0] == fp.WARMUP_FRAMES
    with pytest.raises(RuntimeError):
        fp.sample_intervals(times[:-1])


@pytest.mark.gui
def test_the_inset_is_prepared_in_one_frame_and_drawn_in_the_next(monkeypatch):
    """The robot-camera preview splits its cost: one frame prepares its scene (as the simulation
    is then), the next draws it; it is shown from then on, stamped with the time it shows."""
    app = _app_for_inset()
    sim = app.system.sim
    draws = []
    real = sim.render_prepared
    monkeypatch.setattr(sim, "render_prepared", lambda: draws.append(1) or real())
    app.draw()
    assert app._inset is None and app._inset_pending is not None and not draws
    prepared_at = app._inset_pending[1]
    app.system.advance(1 / 60)
    app.draw()
    assert app._inset is not None and app._inset_pending is None and draws == [1]
    assert app._inset_time == prepared_at
    app.draw()  # not due again yet: nothing new
    assert draws == [1] and app._inset_pending is None
    close_renderer(app.renderer)
    app.system.close()
    app.display.close()
    pygame.quit()


@pytest.mark.gui
@pytest.mark.parametrize("transition", ["restart", "hide panels", "robot camera view", "panel size"])
def test_no_preview_from_before_a_transition_is_shown(transition):
    """After a restart, panels hidden and shown again, the robot-camera view and back, or the
    panels changing size, no robot-camera preview from before is drawn (not even for one frame,
    and not at the wrong size); the next one shows the simulation after the transition."""
    app = _app_for_inset()
    app.draw()
    app.draw()  # a preview is up
    assert app._inset is not None
    app._inset_time = -float("inf")
    app.draw()  # and a new scene prepared (pending)
    assert app._inset_pending is not None
    if transition == "restart":
        app.new_episode(app.seed + 1)
    elif transition == "hide panels":
        app.hud = "none"
        app.draw()
        app.system.advance(1.0)
        app.hud = "compact"
    elif transition == "robot camera view":
        app.view.mode = VIEWS.index("robot camera")
        app.draw()
        app.system.advance(1.0)
        app.view.mode = 0
    else:
        app.hud = "full"
    after = app.system.time
    app.draw()  # the first frame after: nothing old is shown
    assert app._inset is None and app._inset_pending is not None and app._inset_pending[1] >= after
    app.draw()
    w, h = app._inset.get_size()
    rect = (240, 180) if app.hud == "compact" else (320, 240)
    assert (w, h) == rect and app._inset_time >= after
    close_renderer(app.renderer)
    app.system.close()
    app.display.close()
    pygame.quit()


def _key(app, key):
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=key, mod=0, unicode="", scancode=0))
    pygame.event.post(pygame.event.Event(pygame.KEYUP, key=key, mod=0, unicode="", scancode=0))
    return app.handle_events()


@pytest.mark.gui
def test_the_options_button_sets_the_number_of_cats_and_restarts():
    """The window starts with 4 cats. The Options button on the panels opens the options beside
    them (the simulation and driving go on); clicking a number restarts with that many cats (same
    goal); O toggles the options; Esc closes them before it would quit."""
    from robot_env.cats import DEFAULT_CATS
    assert DEFAULT_CATS == 4
    app = App(1000, None, 5, None, 0, cats=DEFAULT_CATS, cat_seed=0)
    app._frame_dt = 1 / 60
    assert app.system.cats.n == 4 and not app.menu.open
    goal = app.episode.task.goal
    app.draw()
    click = lambda pos: pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=pos))  # noqa: E731
    click(app.menu.button.center)
    assert app.handle_events() and app.menu.open
    t = app.system.time
    app.simulate(0.05)
    assert app.system.time > t  # the simulation goes on with the options open
    app.draw()  # lays out the choices
    two = next(rect for rect, i, value in app.menu._targets if value == 2)
    click(two.center)
    assert app.handle_events()
    assert app.cat_count == 2 and app.system.cats.n == 2 and app.system.time == 0.0  # restarted
    assert app.episode.task.goal == goal
    app.draw()
    zero = next(rect for rect, i, value in app.menu._targets if value == 0)
    click(zero.center)
    assert app.handle_events() and app.cat_count == 0 and app.system.cats is None
    assert _key(app, pygame.K_ESCAPE) and not app.menu.open  # Esc closes the options, not the window
    assert _key(app, pygame.K_o) and app.menu.open
    assert _key(app, pygame.K_o) and not app.menu.open
    click(app.menu.button.center)  # the button again
    assert app.handle_events() and app.menu.open
    # hiding the panels (H to "none") closes the options with them; O then shows the panels again
    while app.hud != "none":
        _key(app, pygame.K_h)
    assert not app.menu.open
    assert _key(app, pygame.K_o) and app.menu.open and app.hud == "compact"
    app.draw()
    assert _key(app, pygame.K_ESCAPE) and not app.menu.open  # visible, so Esc closes it
    app._close_all()


def test_the_window_starts_with_four_cats():
    import inspect

    import robot_env.app as A
    from robot_env.cats import DEFAULT_CATS
    assert DEFAULT_CATS == 4 and "default=DEFAULT_CATS" in inspect.getsource(A.main)
