"""All tunable numbers in one place. Units: meters, seconds, radians."""

import math

# Robot geometry
WHEEL_RADIUS = 0.05
WHEEL_SEPARATION = 0.24  # distance between wheel centers
FOOTPRINT_HALF_LENGTH = 0.16  # chassis (0.15) plus a small margin, along x
FOOTPRINT_HALF_WIDTH = 0.14  # outer wheel faces (0.135) plus a small margin, along y
CIRCUMSCRIBED_RADIUS = math.hypot(FOOTPRINT_HALF_LENGTH, FOOTPRINT_HALF_WIDTH)

# Speed limits enforced by the safety layer (any driver, including Shift)
MAX_LINEAR_SPEED = 0.5  # m/s
MAX_ANGULAR_SPEED = 1.5  # rad/s

# Manual driving speeds
MANUAL_NORMAL = (0.30, 1.0)  # (m/s, rad/s)
MANUAL_FAST = (0.50, 1.5)
MANUAL_REVERSE_FACTOR = 0.5  # reverse at half speed: no rear camera (matches back_up)

# Velocity smoother in the drive train (like a real motor driver): limits how fast the
# executed speed may increase. Slowing down and stopping are never delayed.
SMOOTH_ACCEL = 1.5  # m/s^2
SMOOTH_ANG_ACCEL = 6.0  # rad/s^2

# Timing
PHYSICS_DT = 0.002  # 500 Hz (must match world.xml)
CONTROL_PERIOD = 0.02  # 50 Hz: sensing, safety, wheel targets
DECISION_PERIOD = 0.1  # 10 Hz: driver decisions
COMMAND_LIFETIME = 0.3  # a command older than this means stop
SCAN_MAX_AGE = 0.1  # a lidar scan older than this is stale
COLLISION_EVENT_GAP = 0.1  # contact must be absent this long before a new collision counts

# Lidar
LIDAR_RAYS = 36
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
ROOM_HALF_SIZE = 3.0  # inner faces of the walls at +/-3 m
GOAL_RADIUS = 0.3
EPISODE_TIME_LIMIT = 60.0
START_GOAL_MIN_CLEARANCE = 0.5  # from any wall or obstacle surface
START_GOAL_MIN_SEPARATION = 1.5
PLANNING_CLEARANCE = 0.05  # added to the circumscribed radius for path checks
GRID_RESOLUTION = 0.05

# Reward
REWARD_GOAL = 10.0
REWARD_COLLISION = -10.0
REWARD_INTERVENTION = -0.1  # once per continuous intervention event

# Held-out goal seeds (fixed obstacle layout, so these are held-out goals)
HELDOUT_SEEDS = tuple(range(1000, 1010))

# Format versions, recorded with data
OBSERVATION_VERSION = 1
ACTION_VERSION = 1
