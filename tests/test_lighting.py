import mujoco
import numpy as np

from robot_env.layout import region_of
from robot_env.lighting import GL_LIGHTS, Lights
from robot_env.sim import RobotSim


def _names(m):
    return [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_LIGHT, k) or "" for k in range(m.nlight)]


def test_the_world_has_more_lights_than_the_renderer_draws():
    """Why the choice exists: with every light on, the renderer would draw only the first few in
    model order (the window lights) and never a ceiling fitting."""
    m = RobotSim().model
    assert m.nlight > GL_LIGHTS


def test_a_room_view_draws_its_own_fittings_and_windows_within_the_budget():
    sim = RobotSim()
    m = sim.model
    names = _names(m)
    for x, y in ((-3.0, 3.0), (3.0, 3.0), (-3.5, -3.5), (3.0, -3.0), (0.0, 0.0)):
        sim.lights.choose(x, y)
        on = np.flatnonzero(m.light_active)
        assert len(on) <= GL_LIGHTS - (1 if m.vis.headlight.active else 0)
        here = region_of(x, y)
        own = [k for k in range(m.nlight) if sim.lights.region[k] == here and not sim.lights.directional[k]]
        if len(own) + int(sim.lights.directional.sum()) <= len(on):
            assert set(own) <= set(on)  # every light of the room is drawn
        assert any(names[k].startswith("light_") for k in on)  # and at least one ceiling fitting
        assert set(np.flatnonzero(sim.lights.directional)) <= set(on)  # the fills always


def test_the_lights_drawn_change_in_the_renderer_scene():
    sim = RobotSim()
    r = mujoco.Renderer(sim.model, 60, 80)
    try:
        sim.lights.choose(-3.0, 3.0)
        r.update_scene(sim.data)
        drawn = {tuple(round(float(v), 3) for v in r.scene.lights[i].pos) for i in range(r.scene.nlight)}
        office = {tuple(round(float(v), 3) for v in sim.model.light_pos[k]) for k in np.flatnonzero(sim.model.light_active)
                  if not sim.lights.directional[k]}
        assert office <= drawn
    finally:
        r.close()


def test_a_fade_never_draws_more_than_the_budget_and_ends_at_the_new_rooms_lights():
    sim = RobotSim()
    lights, m = sim.lights, sim.model
    budget = lights.budget()
    lights.snap(-3.0, 3.0)
    for _ in range(90):  # 1.5 s at 60 Hz, crossing from the office into the lab
        lights.fade(3.0, 3.0, 1 / 60)
        assert int(m.light_active.sum()) <= budget
        assert np.all(m.light_diffuse <= lights.base + 1e-12)
    assert set(np.flatnonzero(m.light_active)) == set(lights.wanted(3.0, 3.0))
    assert np.allclose(m.light_diffuse[m.light_active > 0], lights.base[m.light_active > 0])


def test_a_fade_changes_the_light_gradually():
    sim = RobotSim()
    lights, m = sim.lights, sim.model
    lights.snap(-3.0, 3.0)
    steps = []
    for _ in range(60):
        before = m.light_diffuse.copy()
        lights.fade(3.0, 3.0, 1 / 60)
        steps.append(np.abs(m.light_diffuse - before).max())
    assert max(steps) <= lights.base.max() * (1 / 60) / 0.3 * 1.01  # no light jumps on or off


def test_shadows_come_only_from_drawn_lights_that_may_cast_them():
    sim = RobotSim()
    m = sim.model
    sim.lights.choose(3.0, 3.0)
    sim.lights.shadows(3.0, 3.0, 2)
    cast = np.flatnonzero(m.light_castshadow)
    assert len(cast) == 2
    assert all(m.light_active[k] and sim.lights.shadowing[k] for k in cast)
    sim.lights.shadows(3.0, 3.0, 0)
    assert not m.light_castshadow.any()


def test_the_robot_camera_image_depends_only_on_the_state():
    """The robot camera picks its lights from the robot's pose, so whatever the window chose
    before, the same state gives the same image."""
    sim = RobotSim()
    sim.reset(-3.0, 3.0, 0.0, (0.0, 0.0))
    first = sim.render_camera((60, 80)).astype(int)
    sim.lights.snap(3.0, -3.0)  # the window looking at another room in between
    again = sim.render_camera((60, 80)).astype(int)
    sim.close()
    assert np.abs(first - again).max() <= 2


def test_a_world_with_few_lights_draws_them_all():
    m = mujoco.MjModel.from_xml_string(
        "<mujoco><worldbody><light pos='0 0 2'/><light pos='1 0 2'/><geom type='plane' size='1 1 .1'/></worldbody></mujoco>")
    lights = Lights(m)
    lights.choose(0.0, 0.0)
    assert m.light_active.all()
