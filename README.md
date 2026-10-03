# Autonomous-Robot-Environment

A 3D simulation of a two-wheeled robot in a closed room, built so that a rapid
decision model
can later drive it. This version is the static foundation: you
drive the robot to a goal by hand, through the same safety layer that any AI
driver will use later.

![Goal reached in the simulator](docs/screenshot_goal.png)


## Setup (Windows, Python 3.11)

In PowerShell, in this folder:

```powershell
.\setup.ps1
```

This creates `.venv\` and installs the pinned packages from `requirements.txt`
(MuJoCo, Gymnasium, Pygame, NumPy, pytest).

## Drive the robot

```powershell
.venv\Scripts\python run_sim.py              # goal seed 1000
.venv\Scripts\python run_sim.py --seed 1003  # another start and goal
```

| Input | Action |
|---|---|
| W/A/S/D or arrow keys | Drive (release to stop). Reverse runs at half speed. |
| Shift | Faster (still within the safety limits) |
| Right mouse button + drag | Drive with the mouse (up = forward, sideways = turn) |
| Space | Emergency brake. Release Space and all drive keys, then press a drive key to go. |
| Left mouse button + drag, mouse wheel | Rotate and zoom the view (chase and orbit views) |
| C | Change view: chase, top, orbit, robot camera |
| R / N | Restart this goal / next goal |
| T | Continue after a collision (manual testing) on or off |
| H | Hide or show the panels |
| F12 | Save a screenshot to `screenshots\` |
| Esc | Quit |

Reach the green goal marker (within 0.3 m) in 60 s without hitting anything.
The status panel shows what you requested, what the safety layer actually
executed, and why ("clearance", "released", and so on).

## Run the tests

```powershell
.venv\Scripts\python -m pytest
```

## How it works

```text
 driver (keyboard now, AI later)
   -> RobotSystem.drive(v, omega)
   -> 50 Hz control tick: lidar scan -> SafetyLayer -> velocity smoother -> wheel motors
   -> 500 Hz MuJoCo physics with contact detection at every step
```

| File | What it does |
|---|---|
| `robot_env/world.xml` | The room, obstacles, goal marker, and robot (MuJoCo) |
| `robot_env/sim.py` | Physics, lidar, camera, wheel encoders, contact detection |
| `robot_env/safety.py` | Safety layer: hard stops, speed limits, clearance prediction |
| `robot_env/system.py` | The one path from any driver to the wheels |
| `robot_env/layout.py` | Room map, seeded start and goal, path check |
| `robot_env/env.py` | Gymnasium environment (`RobotGoalEnv`) for experiments |
| `robot_env/actions.py` | The 5 discrete actions for decision models |
| `robot_env/manual.py`, `robot_env/app.py` | Manual driving and the window |
| `robot_env/config.py` | Every tunable number |

### Gymnasium

```python
from robot_env.env import RobotGoalEnv
env = RobotGoalEnv()
obs, info = env.reset(options={"task_seed": 1000})
obs, reward, terminated, truncated, info = env.step([0.3, 0.0])  # [v m/s, omega rad/s]
```

Observation: lidar ranges (36 rays), lidar validity, velocity estimate from the
wheel encoders, and goal distance and bearing. The goal comes from an **ideal
goal sensor**. That is a stated simulation assumption, since a real robot
would need localization. True pose and velocity are only in `info`, for
evaluation.

## Notes

- The Python environment (`.venv\`) lives in this folder, so Google Drive
  syncs it. It is not in Git; rebuild it with `setup.ps1` on another computer.
- Use one active computer at a time for this folder. Clone from GitHub
  elsewhere.
