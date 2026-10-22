"""Smoke test of the route-input families (hybrid CNN, cfc, dt, mlp, mapcnn): nothing is trained
or logged, outputs go to a temporary folder.

    .venv\\Scripts\\python -m AD.models.hybrid.smoke_route                   (seconds, CPU, no simulation)
    .venv\\Scripts\\python -m AD.models.hybrid.smoke_route --drive --seconds 20 --route online

Without --drive, per family: the widened network equals its base on random inputs (whatever the
route values), a save/load round trip changes nothing, and fit runs on tiny fake episodes; plus the
route source registry and reset_driver. With --drive, each widened family also drives a practice
episode (seed 2003, 4 cats) through its route driver for --seconds simulated seconds and reports
the decision latency (median, p95, share over the 0.1 s period). Run from the main checkout."""

from __future__ import annotations

import argparse
import importlib
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from ...common import tracker
from .route import make_route_source, reset_driver, route_sources, route_spec
from .routekit import Episodes

BASES = {"hybrid": "cnn/dagger_cont_r3", "cfc": "cfc/dagger_cfc_r3", "dt": "dt/dt_noact", "mlp": "mlp/mlp_cont",
         "mapcnn": "mapcnn/dagger_r1"}


def _fake_episodes(rng, n=6, t=40) -> list[dict]:
    eps = []
    for _ in range(n):
        x = rng.uniform(0, 1, (t, 78)).astype(np.float32)
        x[:, 72] = np.linspace(0.5, 0.02, t)  # goal distance falling to arrival
        eps.append({"x": x, "t": (np.arange(t) * 0.1).astype(np.float32),
                    "route": rng.uniform(-1, 1, (t, 4)).astype(np.float32),
                    "cmd": rng.uniform([-0.1, -1], [0.3, 1], (t, 2)).astype(np.float32)})
    return eps


def _outputs(family, net, x78, x82, seq78, seq82, base_path):
    """(widened net on route inputs, base net on the same inputs without route values)."""
    t = torch.from_numpy
    if family == "hybrid":
        from ..cnn.policy import load
        return net(t(x82)), load(base_path)(t(x78))
    if family == "mlp":
        from ..mlp.model import load
        return net(t(x82)), load(Path(base_path))(t(x78))
    if family == "mapcnn":
        from ..mapcnn.policy import load
        m = torch.zeros(len(x78), 3, 32, 32)
        return net(t(x82), m), load(base_path)(t(x78), m)
    if family == "cfc":
        from ..cfc.model import load
        return net(t(seq82))[0], load(base_path)(t(seq78))[0]
    from ..dt.train import make_model, predict  # dt: through predict with a short context
    ck = torch.load(base_path, map_location="cpu", weights_only=False)
    base = make_model(ck["hidden"], ck["layers"], ck["heads"], ck["k"])
    base.load_state_dict(ck["state"])
    k = ck["k"]
    s78 = torch.from_numpy(np.resize(seq78, (2, k, 78)))
    s82 = torch.cat([s78, torch.ones(2, k, 4)], 2)
    a, r, tt, m = torch.zeros(2, k, 2), torch.full((2, k, 1), 0.3), torch.arange(k).repeat(2, 1), torch.ones(2, k)
    return (predict(net.model.eval(), s82, a, r, tt, m, net.actions_in),
            predict(base.eval(), s78, a, r, tt, m, net.actions_in))


def check(families, rng, tmp: Path) -> int:
    fails = 0
    src = make_route_source("online")
    print("route sources:", route_sources(), "| online spec:", route_spec(src))

    class Plain:
        def reset(self):
            self.got = "plain reset"

    class Posed:
        def reset(self, initial_pose=None, **kw):
            self.got = initial_pose
    a, b = Plain(), Posed()
    reset_driver(a, (1.0, 2.0, 0.3))
    reset_driver(b, (1.0, 2.0, 0.3))
    ok = a.got == "plain reset" and b.got == (1.0, 2.0, 0.3)
    fails += not ok
    print("reset_driver:", a.got, "|", b.got, "PASS" if ok else "FAIL")
    Episodes.save(tmp / "e.npz", _fake_episodes(rng, 3, 10))
    lens = [len(e["x"]) for e in Episodes.load([tmp / "e.npz"])]
    fails += lens != [10, 10, 10]
    print("episodes round trip:", lens)
    for family in families:
        t0 = time.time()
        fam = importlib.import_module(f"AD.models.{family}.routed")
        m, r = BASES[family].split("/")
        base_path = str(tracker.AD / "models" / m / "runs" / r / "model.pt")
        net = fam.widen_from(base_path)
        x78 = rng.uniform(0, 1, (5, 78)).astype(np.float32)
        x82 = np.concatenate([x78, rng.uniform(-1, 1, (5, 4)).astype(np.float32)], 1)
        seq78 = rng.uniform(0, 1, (2, 7, 78)).astype(np.float32)
        seq82 = np.concatenate([seq78, rng.uniform(-1, 1, (2, 7, 4)).astype(np.float32)], 2)
        with torch.no_grad():
            wide, base = _outputs(family, net, x78, x82, seq78, seq82, base_path)
            diff = float((wide - base).abs().max())
            path = tmp / family / "model.pt"
            fam.save(net, path, route_spec(src))
            again, _ = _outputs(family, fam.load(path), x78, x82, seq78, seq82, base_path)
            trip = float((again - wide).abs().max())
        fam.fit(fam.load(path), _fake_episodes(rng, 4), _fake_episodes(rng, 2),
                SimpleNamespace(epochs=1, steps=2, batch=4, lr=1e-4, device="cpu"))
        ok = diff < 1e-5 and trip < 1e-5
        fails += not ok
        print(f"{family:7s} widened==base {diff:.1e}  round trip {trip:.1e}  fit ran  weights {fam.params(net)}  "
              f"{'PASS' if ok else 'FAIL'} ({time.time() - t0:.1f} s)", flush=True)
    return fails


def drive(families, tmp: Path, seconds: float, route: str) -> None:
    from robot_env import config as C
    from robot_env.layout import RoomMap
    from robot_env.system import RobotSystem
    for family in families:
        fam = importlib.import_module(f"AD.models.{family}.routed")
        m, r = BASES[family].split("/")
        path = tmp / f"drive_{family}" / "model.pt"
        fam.save(fam.widen_from(tracker.AD / "models" / m / "runs" / r / "model.pt"), path,
                 route_spec(make_route_source(route)))
        driver = fam.Driver(str(path), "cpu", route)
        system = RobotSystem(cats=4, cat_seed=2003)
        task = RoomMap(system.sim.model).sample_task(2003)
        system.reset(*task.start, task.goal)
        reset_driver(driver, task.start)
        lat, outcome = [], "running"
        while system.time < seconds:
            obs = system.observe()
            if obs.goal_distance <= C.GOAL_RADIUS:
                outcome = "reached"
                break
            t0 = time.perf_counter()
            d = driver.decide(obs)
            lat.append(1000 * (time.perf_counter() - t0))
            system.apply(d)
            system.advance(C.DECISION_PERIOD)
        system.close()
        lat = np.array(lat)
        print(f"{family:7s} drive ({route}): {len(lat)} decisions, latency median {np.median(lat):.1f} ms "
              f"p95 {np.percentile(lat, 95):.1f} ms late {np.mean(lat > 100):.3f}, last route {driver.last_route}, {outcome}",
              flush=True)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--families", nargs="*", default=list(BASES), choices=list(BASES))
    p.add_argument("--drive", action="store_true", help="also drive a practice episode per family (simulation)")
    p.add_argument("--seconds", type=float, default=20.0)
    p.add_argument("--route", default="online")
    a = p.parse_args(argv)
    torch.manual_seed(0)
    tmp = Path(tempfile.mkdtemp(prefix="route_smoke_"))
    fails = check(a.families, np.random.default_rng(0), tmp)
    if a.drive:
        drive(a.families, tmp, a.seconds, a.route)
    print("RESULT", "PASS" if not fails else f"FAIL ({fails})", "| temp folder", tmp)
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
