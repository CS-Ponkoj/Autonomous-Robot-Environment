"""GUI smoke tests: the real window, renderer, and event loop, driven by scripted keys.

These show the window works end to end. They do not replace the owner's manual drive."""

import pygame
import pytest

from robot_env.app import App, parse_script


def run_app(tmp_path, **kw):
    shot = tmp_path / "shot.png"
    app = App(kw.pop("seed", 1000), screenshot=shot, **kw)
    summary = app.run()
    return summary, shot


def test_window_renders_and_saves_screenshot(tmp_path):
    summary, shot = run_app(tmp_path, frames=20)
    assert shot.exists() and shot.stat().st_size > 10_000
    assert summary["status"] == "running" and summary["collisions"] == 0


@pytest.mark.parametrize("view", [0, 1, 2, 3])
def test_every_view_renders(tmp_path, view):
    summary, shot = run_app(tmp_path, frames=5, view=view)
    assert shot.exists()


@pytest.mark.parametrize("view, expect_inset", [(0, True), (3, False)])
def test_robot_camera_inset_is_skipped_in_robot_camera_view(tmp_path, monkeypatch, view, expect_inset):
    from robot_env.sim import RobotSim
    calls = []
    original = RobotSim.render_camera

    def counting(self, *args, **kwargs):
        calls.append(1)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(RobotSim, "render_camera", counting)
    run_app(tmp_path, frames=5, view=view)
    assert bool(calls) == expect_inset


def test_parse_script():
    steps = parse_script("W:2;W+A:1;none:0.5;SPACE:0.3")
    assert steps[1].keys == frozenset({pygame.K_w, pygame.K_a}) and steps[2].keys == frozenset()
    assert steps[3].duration == 0.3


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


def test_window_runs_a_non_manual_driver_without_held_keys(tmp_path):
    """Regression: release-to-stop blocked any non-manual driver."""
    app = App(1006, screenshot=tmp_path / "s.png", frames=60, driver=_FakeModelDriver())
    assert not app.manual_driving
    app.run()
    assert app.system.last_decision.command.v == 0.3
    assert app.system.last_result.reasons == () or "clearance" in app.system.last_result.reasons
    assert app.system.sim.true_velocity()[0] > 0.1  # it actually moves


def test_space_still_brakes_a_non_manual_driver(tmp_path):
    app = App(1006, screenshot=tmp_path / "s.png", frames=90, driver=_FakeModelDriver(),
              script=parse_script("none:0.6;SPACE:1.0"))
    app.run()
    assert app.system.last_result.reasons == ("emergency_brake",)
    assert abs(app.system.sim.true_velocity()[0]) < 0.02


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


def test_scripted_keys_move_then_release_stops(tmp_path):
    app = App(1005, screenshot=None, frames=300, script=parse_script("D:1.5;W:1.5;none:1.2"))  # about 4.2 s of script
    x0, y0, _ = app.system.sim.true_pose()
    app.run()
    x1, y1, _ = app.system.sim.true_pose()
    assert ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5 > 0.2  # it really moved
    assert abs(app.system.sim.true_velocity()[0]) < 0.02  # and released keys stopped it
    assert app.system.collisions == 0


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


def test_restart_and_next_goal_keep_the_selected_level():
    app = _open_app()
    _key_event(app, pygame.KEYDOWN, pygame.K_MINUS)
    level = app.input.level
    app.new_episode(app.seed)
    app.new_episode(app.seed + 1)
    assert app.input.level == level
    app.run()


def test_status_line_shows_selected_and_effective_levels():
    from robot_env import config as C
    app = _open_app()
    n = len(C.SPEED_LEVELS)
    assert app.speed_level_text().startswith(f"speed level {C.DEFAULT_SPEED_LEVEL + 1}/{n}: ")
    _key_event(app, pygame.KEYDOWN, pygame.K_RSHIFT)
    assert f"Shift: level {n}" in app.speed_level_text()
    app.run()


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
