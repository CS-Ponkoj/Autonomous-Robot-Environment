"""The cats' 100 Hz samples with per-physics-step interpolation, against a 2 kHz reference (cat
motion computed at every physics step). Development gate.

Scenarios (scripted, so both runs follow the same paths): a cat walking into the stopped robot,
the robot driving into a still cat, a cat walking past the robot while turning, a cat that sees
the robot raising its tail (the alert lift starting mid-run) as it walks into it, and the robot
driving into a cat's low tail and into its raised tail. For each: the largest difference of a
collider endpoint over the run (gate: 10 mm for the body and head, 10 mm for the tail; 20 mm for
the legs, whose steps can only start at a sample boundary, so a paw swinging at about 1.5 m/s
can be up to one 10 ms sample, 15 mm, behind the reference), the time of the first
contact (gate: within 10 ms), the number of contacts and who moved into whom (gate: the same),
no tunnelling (the robot never moves more than 2 mm while a cat touches it), and the peak contact
force and impulse (gate: within 10% of each other, or within 15 N and 1 N s). Why that floor: when
the robot drives into a cat both runs see the same contact (they match exactly), but when a
cat's swinging paw touches the robot, the force is set by how far the paw got within one 0.5 ms
physics step (it moves up to 0.75 mm in one), so two correct runs differ there; such a touch is
light (well under 15 N and 1 N s), and what it does to the robot is gated above (2 mm).

    .venv\\Scripts\\python tools\\interp_check.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from robot_env import cats as K  # noqa: E402
from robot_env import config as C  # noqa: E402
from robot_env.system import RobotSystem  # noqa: E402

NAMES = [c.name for c in K.colliders()]
BODY = np.array([n in ("torso_rear", "torso_front", "head") for n in NAMES])
TAIL = np.array([n.startswith("tail") for n in NAMES])
LEGS = ~BODY & ~TAIL
LIMITS = {"body": 0.010, "tail": 0.010, "legs": 0.020}  # m
FORCE_REL, FORCE_FLOOR, IMPULSE_FLOOR = 0.10, 15.0, 1.0  # fraction, N, N s (see above)
SCENARIOS = {
    # name: (robot pose, robot command (v, w), cat pose (x, y, yaw), cat command (v, w), seconds,
    #        the cat sees the robot (its tail rises when the robot is within ALERT_DISTANCE))
    "cat walks into the stopped robot": ((-1.5, 0.0, 0.0), (0.0, 0.0), (-0.5, 0.0, math.pi), (0.4, 0.0), 3.0, False),
    "robot drives into a still cat": ((-2.0, 0.0, 0.0), (0.3, 0.0), (-1.0, 0.0, math.pi / 2), (0.0, 0.0), 4.0, False),
    "cat turns past the robot": ((-1.5, 0.0, 0.0), (0.0, 0.0), (-0.6, -0.45, math.pi), (0.35, 0.5), 3.0, False),
    "alert cat raises its tail walking into the robot": ((-2.0, 0.0, 0.0), (0.0, 0.0), (0.0, 0.0, math.pi), (0.4, 0.0), 6.0, True),
    "robot drives into a low tail": ((-2.0, 0.0, 0.0), (0.3, 0.0), (-0.95, 0.0, 0.0), (0.0, 0.0), 4.0, False),
    "robot drives at a cat with its tail raised": ((-2.0, 0.0, 0.0), (0.3, 0.0), (-0.95, 0.0, 0.0), (0.0, 0.0), 4.0, True),
}


def run(name: str, reference: bool) -> dict:
    robot, rcmd, cat, ccmd, seconds, sees = SCENARIOS[name]
    saved = (K.ANIM_PERIOD, K.ROOT_SUBSTEPS)
    if reference:
        K.ANIM_PERIOD, K.ROOT_SUBSTEPS = C.PHYSICS_DT, 1  # a new sample every physics step
    try:
        from robot_env.safety import SafetyLayer
        # fault injection: no clearance filter, so the robot really drives into the cat
        s = RobotSystem(safety=SafetyLayer(clearance_enabled=False), cats=1, cat_seed=0)
        s.reset(*robot, (4.5, 4.5))
        h = s.cats
        # scripted: no behaviour, no planner, no clearance rules (contacts are what is measured)
        h.pose_ok = lambda *a, **k: True
        h.tick = lambda **k: None

        def straight(c):
            c.cmd_v, c.cmd_w, c.cmd_lat = ccmd[0], ccmd[1], 0.0
        h._plan = straight
        h.place(0, *cat, v=ccmd[0], state="walk")
        h.cats[0].sees_robot = sees  # (behaviour is off: fixed for the run)
        lifts = []
        parts = set()
        ends, times = [], []
        onset = None
        robot_start = None
        moved_in_contact = 0.0
        while s.time < seconds - 1e-9:
            if rcmd != (0.0, 0.0):
                s.drive(*rcmd)
            s.advance(C.PHYSICS_DT)
            ends.append(h.col_world[0].copy())
            times.append(s.time)
            lifts.append(h.cats[0].animator.tail_lift)
            m = s.sim.model
            for touching in s.sim.cat_contacts_now().values():
                for c, sign in touching:
                    g = s.sim.data.contact.geom[c][1 if sign > 0 else 0]
                    parts.add(NAMES[int(m.body(int(m.geom_bodyid[g])).name.split("_c", 1)[1])])
            touching = h._cat_open if hasattr(h, "_cat_open") else {}
            if s.cat_contacts and onset is None:
                onset = s.time
                robot_start = np.array(s.sim.true_pose()[:2])
            if onset is not None and s.cat_contacts and robot_start is not None and rcmd == (0.0, 0.0):
                moved_in_contact = max(moved_in_contact, float(np.linalg.norm(np.array(s.sim.true_pose()[:2]) - robot_start)))
        events = list(s.cat_contact_events)
        s.close()
        return {"ends": np.array(ends), "times": np.array(times), "onset": onset, "contacts": len(events),
                "initiators": [e.get("initiator") for e in events],
                "force": max([e.get("peak_force_n", 0.0) for e in events] or [0.0]),
                "impulse": max([e.get("impulse_ns", 0.0) for e in events] or [0.0]),
                "moved_in_contact": moved_in_contact, "lift": max(lifts),
                "parts": sorted(parts)}
    finally:
        K.ANIM_PERIOD, K.ROOT_SUBSTEPS = saved


def main() -> int:
    ok = True
    for name in SCENARIOS:
        prod, ref = run(name, False), run(name, True)
        n = min(len(prod["ends"]), len(ref["ends"]))
        # compare up to the first contact (after it, a frozen cat and a moving one differ by design)
        stop = n
        for r in (prod, ref):
            if r["onset"] is not None:
                stop = min(stop, int(np.searchsorted(r["times"], r["onset"])))
        per = np.abs(prod["ends"][:stop] - ref["ends"][:stop]).max(axis=(0, 2, 3)) if stop else np.zeros(len(BODY))
        worst = {"body": float(per[BODY].max()), "tail": float(per[TAIL].max()), "legs": float(per[LEGS].max())}
        print(f"  {name}, each collider (mm):", {n: round(1000 * float(v), 1) for n, v in zip(NAMES, per)})
        onset_ok = (prod["onset"] is None) == (ref["onset"] is None) and (
            prod["onset"] is None or abs(prod["onset"] - ref["onset"]) <= 0.010 + 1e-9)
        same = prod["contacts"] == ref["contacts"] and prod["initiators"] == ref["initiators"] and prod["parts"] == ref["parts"]
        tunnel = prod["moved_in_contact"] <= 0.002 and ref["moved_in_contact"] <= 0.002
        lifted = (prod["lift"] > 0.5) == SCENARIOS[name][5] and (ref["lift"] > 0.5) == SCENARIOS[name][5]
        force_ok = abs(prod["force"] - ref["force"]) <= max(FORCE_REL * max(prod["force"], ref["force"]), FORCE_FLOOR)
        impulse_ok = abs(prod["impulse"] - ref["impulse"]) <= max(FORCE_REL * max(prod["impulse"], ref["impulse"]),
                                                                  IMPULSE_FLOOR)
        passed = (all(worst[k] <= LIMITS[k] for k in LIMITS) and onset_ok and same and tunnel and lifted
                  and force_ok and impulse_ok)
        ok &= passed
        print(f"{name}: endpoint difference body {1000 * worst['body']:.2f} mm, tail {1000 * worst['tail']:.2f} mm, "
              f"legs {1000 * worst['legs']:.2f} mm; tail lift {prod['lift']:.2f} / {ref['lift']:.2f} rad; "
              f"onset {prod['onset']} vs {ref['onset']}; parts touched {prod['parts']} vs {ref['parts']}; "
              f"contacts {prod['contacts']} vs {ref['contacts']} initiators {prod['initiators']} vs {ref['initiators']}; "
              f"robot moved while touched {1000 * prod['moved_in_contact']:.2f} / {1000 * ref['moved_in_contact']:.2f} mm; "
              f"peak force {prod['force']:.2f} vs {ref['force']:.2f} N, impulse {prod['impulse']:.4f} vs {ref['impulse']:.4f} N s: "
              f"{'PASS' if passed else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
