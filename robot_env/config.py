"""All tunable numbers in one place. Units: meters, seconds, radians."""

import math

# Robot geometry
WHEEL_RADIUS = 0.05
WHEEL_SEPARATION = 0.24  # distance between wheel centers
FOOTPRINT_HALF_LENGTH = 0.16  # chassis (0.15) plus a small margin, along x
FOOTPRINT_HALF_WIDTH = 0.14  # outer wheel faces (0.135) plus a small margin, along y
CIRCUMSCRIBED_RADIUS = math.hypot(FOOTPRINT_HALF_LENGTH, FOOTPRINT_HALF_WIDTH)

# Speed levels for manual driving, (m/s, rad/s), slowest first. This tuple is the single
# source of truth: only validated levels are listed, and the speed cap, the +/- keys, Shift,
# the window, the command line, and the Gym action space all follow it.
SPEED_LEVELS = ((0.20, 0.8), (0.30, 1.0), (0.50, 1.3), (0.70, 1.5), (1.00, 1.5))
DEFAULT_SPEED_LEVEL = 1  # index into SPEED_LEVELS: shown as level 2, 0.30 m/s
MANUAL_REVERSE_FACTOR = 0.5  # reverse at half speed: no rear camera (matches back_up)

# Speed limits enforced by the safety layer (any driver)
MAX_LINEAR_SPEED = max(v for v, _ in SPEED_LEVELS)  # m/s: the fastest exposed level
MAX_ANGULAR_SPEED = 1.5  # rad/s

# Velocity smoother in the drive train (like a real motor driver), applied before every
# physics step: limits how fast the executed speed may rise and, for ordinary slowing, fall.
# Hard safety stops bypass it (the motor target becomes zero at once).
SMOOTH_ACCEL = 1.5  # m/s^2
SMOOTH_ANG_ACCEL = 6.0  # rad/s^2
SMOOTH_DECEL = 3.0  # m/s^2, ordinary slowing down (faster than SAFETY_DECEL, so predictions stay conservative)
SMOOTH_ANG_DECEL = 10.0  # rad/s^2 (faster than SAFETY_ANG_DECEL)

# Timing
PHYSICS_DT = 0.0005  # 2 kHz (must match world.xml); 0.002 let the tire contact creep
CONTROL_PERIOD = 0.02  # 50 Hz: sensing, safety, wheel targets
DECISION_PERIOD = 0.1  # 10 Hz: driver decisions
COMMAND_LIFETIME = 0.3  # a command older than this means stop
SCAN_MAX_AGE = 0.1  # a lidar scan older than this is stale
COLLISION_EVENT_GAP = 0.1  # contact must be absent this long before a new collision counts

# Lidar: one horizontal plane, 1 degree spacing (typical of real 2D lidars). With 36 rays
# (10 degrees) a 2.4 cm pole could hide between rays and be hit.
LIDAR_RAYS = 360
LIDAR_RANGE = 5.0
LIDAR_ANGLES = tuple(-math.pi + i * 2 * math.pi / LIDAR_RAYS for i in range(LIDAR_RAYS))

# Safety clearance model, derived from the step 5 motion check (measured on an
# empty track, see tests/test_motion.py which re-checks these stay conservative):
#   linear: 90% of 0.3-0.5 m/s in 0.14-0.16 s (avg accel 1.9-2.8 m/s^2);
#           stops from 0.3 m/s in 1.5 cm (effective decel ~3 m/s^2 incl. delay)
#   angular: 90% of 1.0-1.5 rad/s in 0.08-0.10 s (avg 11-13.5 rad/s^2);
#           stops from 1.0 rad/s within 0.03 rad
# The predictor assumes faster acceleration and slower braking than measured,
# so it overestimates travel (safe side).
SAFETY_BUFFER = 0.05  # minimum gap kept between footprint and obstacles
SAFETY_DELAY = 0.04  # control period plus actuation delay
SAFETY_ACCEL = 6.0  # m/s^2 (measured avg <= 2.8, peak higher: covers 0.5 m/s start-up)
SAFETY_DECEL = 2.0  # m/s^2 (measured effective ~3)
SAFETY_ANG_ACCEL = 15.0  # rad/s^2 (measured avg <= 13.5)
SAFETY_ANG_DECEL = 8.0  # rad/s^2 (measured effective ~15)
SAFETY_HOLD = 0.1  # time the requested command is assumed held (one decision)
SAFETY_PREDICT_DT = 0.02
SAFETY_CONSIDER_RADIUS = 1.5  # ignore obstacle points farther than this

# Room and task
FLOOR_HALF_SIZE = 5.0  # inner faces of the outer walls at +/-5 m (tools/build_world.py)
WALL_HEIGHT = 2.4
# Regions of the office floor: (x_min, x_max, y_min, y_max). Every task's start and goal
# are in different regions, so each task crosses at least one doorway.
ROOMS = {
    "office": (-5.0, -0.05, 0.8, 5.0),
    "lab": (0.05, 5.0, 0.8, 5.0),
    "storage": (-5.0, -0.05, -5.0, -0.8),
    "reception": (0.05, 5.0, -5.0, -0.8),
    "corridor": (-5.0, 5.0, -0.7, 0.7),
}
GOAL_RADIUS = 0.3
# Episode limit: at least twice the worst estimated travel time over 1,000 sampled tasks
# on the office floor (re-measured after every layout change; see the plan). With the
# furnished floor the worst estimate is 63.0 s (seed 729), so 130 s keeps every task
# within half. The estimate is path length and turning, not a measured drive time.
EPISODE_TIME_LIMIT = 130.0
WORST_TRAVEL_SEED = 729  # regression: the worst seed of the last 1,000-seed sweep
START_GOAL_MIN_CLEARANCE = 0.5  # from any wall or obstacle surface
START_GOAL_MIN_SEPARATION = 1.5
PLANNING_CLEARANCE = 0.10  # added to the circumscribed radius for path checks: twice the
# safety buffer, so the robot can turn in place anywhere on a path (0.05 put the corners
# exactly on the buffer at doorways). Needs 2 x 0.31 m + one grid cell < 0.9 m door width.
GRID_RESOLUTION = 0.05
OVERHEAD_CLEARANCE = 0.3  # obstacles entirely above this height (door headers, desk tops) do not block the robot

# Reward
REWARD_GOAL = 10.0
REWARD_COLLISION = -10.0
REWARD_INTERVENTION = -0.1  # once per continuous intervention event

# Held-out goal seeds (fixed obstacle layout, so these are held-out goals)
HELDOUT_SEEDS = tuple(range(1000, 1010))

# Format versions, recorded with data. Bump them whenever the format changes, so datasets
# from different formats are never mixed.
# Observation 2: 360 lidar rays (was 36). Action 2: [v, omega] bounds +/-1.0 m/s (was 0.5).
OBSERVATION_VERSION = 2
ACTION_VERSION = 2
