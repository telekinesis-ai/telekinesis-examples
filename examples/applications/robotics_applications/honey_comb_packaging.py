'''
Prompt: I have a UR10e and an RG6 gripper, I want to do a repackaging task where
the parts are placed in a rectangular grid and need to be placed into another
grid where there is a fixed offset on the x axis and the y axis and every other
row is offset from the previous row. The first row has n slots, second m third n
etc. Every other row is identical. When picking up the parts do not open the
gripper all the way as the parts are close. start and end the program at a
home position
'''

# Telekinesis pipeline: UR10e + OnRobot RG6 repackaging pick-and-place
# Picks from a regular rows x cols source grid and places into a staggered destination
# grid with alternating row slot counts (n, m, n, m, ...) offset from the source.
#
# Key constraint: gripper uses a PARTIAL open width during picks (source parts are
# closely packed); normal open is fine between moves / after place.

import numpy as np
from loguru import logger

from telekinesis.synapse.robots.manipulators.universal_robots import UniversalRobotsUR10E
from telekinesis.synapse.tools.parallel_grippers.onrobot import OnRobotRG6
# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ROBOT_IP = "192.168.1.2"
GRIPPER_IP = "192.168.1.1"

# Home pose [x, y, z, rx, ry, rz] in meters + degrees (tool pointing down).
HOME_POSE = [0.00, 0.8000, 0.400, 180.0, 0.0, 90.0]

# Common pick/place orientation (tool pointing down).
TOOL_RX, TOOL_RY, TOOL_RZ = 180.0, 0.0, 45.0

# Z heights (meters)
Z_APPROACH = 0.350   # safe travel height above grids
Z_PICK = 0.21   # contact height at source
Z_PLACE = 0.23   # contact height at destination

# --- Source grid (regular rows x cols) ---
SRC_ORIGIN_XY = (0.0235, 0.86715)   # (x0, y0) of source slot (0,0) in meters
SRC_ROWS = 1
SRC_COLS = 1
SRC_DX = 0.045                   # spacing along x between columns
SRC_DY = -0.045                       # spacing along y between rows

# --- Destination grid (staggered, alternating n/m row slot counts) ---
# Destination origin is offset from source origin by a fixed (x, y)
# offset.   # (dx, dy) from source origin to destination origin
DST_ORIGIN_XY = (-0.3825, 0.87011)

DST_N = 5                          # slots in odd rows (row index 0, 2, 4, ...)
DST_M = 4                          # slots in even rows (row index 1, 3, 5, ...)
DST_DX = 0.0297                     # spacing along x between slots within a row
DST_DY = -0.023275                     # spacing along y between rows
DST_STAGGER_X = DST_DX / 2.0       # zigzag x-shift applied to even rows

# --- Gripper widths (mm; RG6 stroke 0..160 mm) ---
GRIP_OPEN_FULL_MM = 120.0   # normal open between moves / after place
GRIP_OPEN_PARTIAL_MM = 60.0    # PARTIAL open used when approaching/releasing at source
GRIP_CLOSE_MM = 0.0     # close on the part (hardware will stop on contact)
GRIP_FORCE_N = 40.0    # RG6 max is 120 N

# Motion params
MOVE_SPEED = 0.3
MOVE_ACCEL = 0.5

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def source_slot_pose(row: int, col: int, z: float) -> list[float]:
    x = SRC_ORIGIN_XY[0] + col * SRC_DX
    y = SRC_ORIGIN_XY[1] + row * SRC_DY
    return [x, y, z, TOOL_RX, TOOL_RY, TOOL_RZ]


def destination_slot_pose(flat_index: int, z: float) -> list[float]:
    """
    Map a flat slot index to an (x, y) pose in the staggered destination grid
    with alternating row slot counts (N, M, N, M, ...).
    Odd rows (0, 2, ...) have DST_N slots; even rows (1, 3, ...) have DST_M slots
    and are x-staggered by DST_STAGGER_X.
    """
    remaining = flat_index
    row = 0
    while True:
        row_slots = DST_N if (row % 2 == 0) else DST_M
        if remaining < row_slots:
            col = remaining
            break
        remaining -= row_slots
        row += 1

    x_offset = 0.0 if (row % 2 == 0) else DST_STAGGER_X
    x = DST_ORIGIN_XY[0] + col * DST_DX + x_offset
    y = DST_ORIGIN_XY[1] + row * DST_DY
    return [x, y, z, TOOL_RX, TOOL_RY, TOOL_RZ]


def above(pose: list[float], z: float) -> list[float]:
    p = list(pose)
    p[2] = z
    return p

# ---------------------------------------------------------------------------
# Connect hardware
# ---------------------------------------------------------------------------


robot = UniversalRobotsUR10E()
robot.connect(ROBOT_IP)

gripper = OnRobotRG6()
# OnRobot RG6 uses Modbus TCP per synapse.tools.parallel_grippers.onrobot
gripper.connect(ip=GRIPPER_IP, protocol="MODBUS_TCP")
gripper.set_unit("position", "mm")
gripper.set_unit("force", "N")

# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

try:
    # 1) Start at home with gripper at normal open
    gripper.move(position=GRIP_OPEN_FULL_MM, force=GRIP_FORCE_N, asynchronous=False)
    robot.set_cartesian_pose(HOME_POSE, speed=MOVE_SPEED, acceleration=MOVE_ACCEL)

    # 2) Iterate source grid row-major; place into staggered destination by flat index
    flat_index = 0
    for r in range(SRC_ROWS):
        for c in range(SRC_COLS):
            pick_pose = source_slot_pose(r, c, Z_PICK)
            pick_above = above(pick_pose, Z_APPROACH)

            place_pose = destination_slot_pose(flat_index, Z_PLACE)
            place_above = above(place_pose, Z_APPROACH)

            logger.info(f"[{flat_index}] pick src(r={r},c={c}) -> place flat={flat_index}")

            # --- PICK ---
            # Partial open BEFORE entering the tight source region
            gripper.move(position=GRIP_OPEN_PARTIAL_MM,
                         force=GRIP_FORCE_N, asynchronous=False)

            robot.set_cartesian_pose(pick_above, speed=MOVE_SPEED, acceleration=MOVE_ACCEL)
            robot.set_cartesian_pose(pick_pose, speed=MOVE_SPEED, acceleration=MOVE_ACCEL)

            # Grasp
            gripper.move(position=GRIP_CLOSE_MM, force=GRIP_FORCE_N, asynchronous=False)

            # Retreat while still at partial width (avoid hitting neighbors on exit)
            robot.set_cartesian_pose(pick_above, speed=MOVE_SPEED, acceleration=MOVE_ACCEL)

            # --- PLACE ---
            robot.set_cartesian_pose(place_above, speed=MOVE_SPEED, acceleration=MOVE_ACCEL)
            robot.set_cartesian_pose(place_pose, speed=MOVE_SPEED, acceleration=MOVE_ACCEL)

            # Normal open on release (destination has room due to stagger/offset)
            gripper.move(position=GRIP_OPEN_FULL_MM,
                         force=GRIP_FORCE_N, asynchronous=False)

            robot.set_cartesian_pose(place_above, speed=MOVE_SPEED, acceleration=MOVE_ACCEL)

            flat_index += 1

    # 3) Return to home
    robot.set_cartesian_pose(HOME_POSE, speed=MOVE_SPEED, acceleration=MOVE_ACCEL)

finally:
    # Clean disconnect
    try:
        gripper.disconnect()
    except Exception as e:
        logger.warning(f"Gripper disconnect failed: {e}")
    try:
        robot.disconnect()
    except Exception as e:
        logger.warning(f"Robot disconnect failed: {e}")
