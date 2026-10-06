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

# Forward depth sensor: a ray frustum from the robot camera's mount, pitched down so its lowest
# rows meet the floor just ahead of the bumper. The lidar plane (0.136 m) is at the robot's own
# top, so anything shorter is invisible to it; the depth sensor sees it (ahead only: the sides
# and the rear stay lidar-only).
DEPTH_COLS, DEPTH_ROWS = 64, 36
DEPTH_HFOV, DEPTH_VFOV = math.radians(80.0), math.radians(50.0)
DEPTH_PITCH = math.radians(20.0)  # tilted down
DEPTH_ORIGIN = (0.158, 0.0, 0.083)  # m in the robot body frame (the camera mount); the body is 0.0505 m up
DEPTH_BODY_Z = 0.0505  # m: the body frame's height above the floor at rest
DEPTH_MIN, DEPTH_MAX = 0.15, 3.0  # m: closer is invalid (blind), farther is no return
DEPTH_PERIOD = 1.0 / 15.0  # s between frames (a frame costs about 2.6 ms in the furnished world with
                           # cats; 15 Hz keeps the window at its frame rate, and a frame is never older
                           # than DEPTH_MAX_AGE)
DEPTH_MAX_AGE = 0.10  # s: older depth cannot authorise forward motion
ROBOT_TOP = 0.1355  # m: the robot's collision height (its tallest solid part)
DEPTH_FLOOR = 0.02  # m: returns lower than this are the floor
DEPTH_VOXEL = 0.02  # m: obstacle points are kept one per cell of this size

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
# Doorway centres (x, y) on the office floor, mirrored from tools/build_world.py (a test checks)
DOORS = ((-2.5, 0.75), (2.5, 0.75), (-3.0, -0.75), (2.0, -0.75), (0.0, 3.5))
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
# The held-out tasks themselves, frozen: seed -> ((start x, y, yaw), (goal x, y)), exactly as
# RoomMap.sample_task drew them in the world before the office was furnished (revision 9ba1e37;
# the world was the same from 1789b77). Sampling is rejection sampling, so a furniture change
# could otherwise re-draw some of them and silently change the evaluation set; RoomMap.sample_task
# returns these for these seeds and checks they are still valid tasks in the current world.
HELDOUT_TASKS = {
    1000: ((-1.966089612223335, 2.2831339697243465, 2.9830288113203025), (1.3734388427585031, 3.5892677587361845)),
    1001: ((-2.1054866881776273, 4.1069808661445535, 2.4456676346702695), (2.555169089225979, -3.5232826094209306)),
    1002: ((3.8596630769843596, -1.368781184639567, -3.123839194612621), (-1.9523109558208036, -4.211362513926343)),
    1003: ((-1.310514674887072, 0.1693768133844511, 0.5832241313342283), (3.880769213382914, -3.1862438585400517)),
    1004: ((-2.031753825514747, 2.9248829534735696, -2.887087219091781), (1.2757521379645107, -2.5676221839310025)),
    1005: ((1.0349775977552635, -0.05295767411052754, 1.063356772585264), (-0.7740605511970582, 3.4631467699145553)),
    1006: ((1.573685493493346, -0.09948933710079544, 0.3199358468373763), (-0.9103363587469744, 3.485729774899081)),
    1007: ((-3.858975562852028, 2.843823106601489, 1.9413443971231272), (2.847991851117346, -0.13611592399197647)),
    1008: ((-2.9947252585084314, 2.8174567295590123, -0.6353958856114446), (4.348694467288865, 3.0984811950142603)),
    1009: ((2.355713300867432, -0.10412983434553702, -3.0697582416171207), (-2.1394172851395608, -1.5970020373874054)),
}

# Format versions, recorded with data. Bump them whenever the format changes, so datasets
# from different formats are never mixed.
# Observation 2: 360 lidar rays (was 36). Action 2: [v, omega] bounds +/-1.0 m/s (was 0.5).
OBSERVATION_VERSION = 3  # 3: the forward depth frame (depth, depth_valid)
ACTION_VERSION = 2
