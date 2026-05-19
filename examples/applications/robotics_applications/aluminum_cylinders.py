'''
Prompt: I have a UR10e and an RG6 gripper. I have parts vertically placed in a grid that need to
be picked and placed horizontally in a new grid (requires a -90-degree flip between pick and place around the y axis).
Add an optional intermediate joint pose before the flip. Add logging at each step.
'''

# Pick-and-place pipeline: UR10e + OnRobot RG6
# Picks parts from a vertical grid (TCP pointing down) and places them
# in a horizontal grid (TCP rotated 90 deg) — flipping orientation during transfer.

import time
import numpy as np
from loguru import logger

from telekinesis.synapse.robots.manipulators.universal_robots import UniversalRobotsUR10E
from telekinesis.synapse.tools.parallel_grippers.onrobot import OnRobotRG6

# ============================================================
# Constants
# ============================================================

ROBOT_IP = "192.168.1.2"
GRIPPER_IP = "192.168.1.1"
GRIPPER_PROTOCOL = "MODBUS_TCP"

# Orientations (Euler XYZ degrees)
PICK_RPY_DEG = [180, 0, 90]                    # TCP pointing down
PLACE_RPY_DEG = [0.0, -90.0, -90.0]            # HARD-CODED literal — do not compute

# Optional joint-space waypoint to force a specific IK branch during the flip.
# Set to None to skip.
INTERMEDIATE_JOINT_POSE_DEG = [90, -100.0, -115.0, -141.0, -90.0, 90.0]

# Home / safe joint pose (deg)
HOME_JOINT_POSE_DEG = [90.0, -90.0, -90.0, -90.0, 90.0, 0.0]

# ---- Source (vertical) grid: parts standing upright, gripper picks from above ----
PICK_GRID_ROWS = 4
PICK_GRID_COLS = 5
PICK_ORIGIN_XYZ = [0.0235, 0.86715, 0.21]      # base position of first part (m)
PICK_ROW_PITCH = -0.0675                          # spacing along Y (m)
PICK_COL_PITCH = 0.0675                          # spacing along X (m)
PICK_APPROACH_OFFSET_Z = 0.100                  # approach/retreat height above pick (m)

# ---- Destination (horizontal) grid: parts laid on their side ----
PLACE_GRID_ROWS = 5
PLACE_GRID_COLS = 4
PLACE_ORIGIN_XYZ = [-0.36323, 0.6682, 0.08]      # base position of first slot (m)
PLACE_ROW_PITCH = -0.0601
PLACE_COL_PITCH = 0.046
PLACE_APPROACH_OFFSET_Z = 0.100                 # approach/retreat height above place (m)

# Motion parameters
CART_SPEED = 0.65          # m/s
CART_ACCEL = 0.85           # m/s^2
JOINT_SPEED_DEG = 45.0     # deg/s
JOINT_ACCEL_DEG = 90.0     # deg/s^2

# Gripper parameters
GRIPPER_OPEN_WIDTH_MM = 60.0
GRIPPER_CLOSE_WIDTH_MM = 10.0
GRIPPER_FORCE_N = 40.0


# ============================================================
# Helper functions
# ============================================================

def generate_pick_poses():
    """Generate list of pick poses [x, y, z, rx, ry, rz] for the vertical source grid."""
    poses = []
    for r in range(PICK_GRID_ROWS):
        for c in range(PICK_GRID_COLS):
            x = PICK_ORIGIN_XYZ[0] + c * PICK_COL_PITCH
            y = PICK_ORIGIN_XYZ[1] + r * PICK_ROW_PITCH
            z = PICK_ORIGIN_XYZ[2]
            poses.append([x, y, z] + list(PICK_RPY_DEG))
    return poses


def generate_place_poses():
    """Generate list of place poses [x, y, z, rx, ry, rz] for the horizontal destination grid."""
    poses = []
    for r in range(PLACE_GRID_ROWS):
        for c in range(PLACE_GRID_COLS):
            x = PLACE_ORIGIN_XYZ[0] + c * PLACE_COL_PITCH
            y = PLACE_ORIGIN_XYZ[1] + r * PLACE_ROW_PITCH
            z = PLACE_ORIGIN_XYZ[2]
            # PLACE_RPY_DEG is hard-coded literal, not derived
            poses.append([x, y, z] + list(PLACE_RPY_DEG))
    return poses


def _approach_pose(pose, dz):
    """Return copy of pose with z offset by dz (leaves orientation unchanged)."""
    p = list(pose)
    p[2] = p[2] + dz
    return p


def pick_part(robot, gripper, pick_pose):
    """Execute pick sequence for a single part."""
    approach = _approach_pose(pick_pose, PICK_APPROACH_OFFSET_Z)

    logger.info("[1] Move to pick approach")
    logger.debug(f"    approach pose = {approach}")
    robot.set_cartesian_pose(approach, speed=CART_SPEED, acceleration=CART_ACCEL)
    logger.success("[1] At pick approach")

    logger.info("[2a] Open gripper")
    gripper.move(position=GRIPPER_OPEN_WIDTH_MM, force=GRIPPER_FORCE_N)
    logger.success("[2a] Gripper opened")

    logger.info("[2b] Move down to pick pose")
    logger.debug(f"    pick pose = {pick_pose}")
    robot.set_cartesian_pose(pick_pose, speed=CART_SPEED, acceleration=CART_ACCEL)
    logger.success("[2b] At pick pose")

    logger.info("[2c] Close gripper to grasp")
    gripper.move(position=GRIPPER_CLOSE_WIDTH_MM, force=GRIPPER_FORCE_N)
    logger.success("[2c] Grasped")

    logger.info("[2d] Retreat upward from pick pose")
    robot.set_cartesian_pose(approach, speed=CART_SPEED, acceleration=CART_ACCEL)
    logger.success("[2d] Retreated from pick")


def place_part(robot, gripper, place_pose):
    """Execute place sequence for a single part.

    Includes step [3] (intermediate joint-space waypoint) and step [4]
    (move to place approach) before descending.
    """
    approach = _approach_pose(place_pose, PLACE_APPROACH_OFFSET_Z)

    if INTERMEDIATE_JOINT_POSE_DEG is not None:
        logger.info("[3] Move to intermediate joint pose (IK-branch control during flip)")
        logger.debug(f"    intermediate q = {INTERMEDIATE_JOINT_POSE_DEG}")
        robot.set_joint_positions(
            INTERMEDIATE_JOINT_POSE_DEG,
            speed=JOINT_SPEED_DEG,
            acceleration=JOINT_ACCEL_DEG,
        )
        logger.success("[3] At intermediate joint pose")
    else:
        logger.info("[3] Skipped (INTERMEDIATE_JOINT_POSE_DEG is None)")

    logger.info("[4] Move to place approach")
    logger.debug(f"    approach pose = {approach}")
    robot.set_cartesian_pose(approach, speed=CART_SPEED, acceleration=CART_ACCEL)
    logger.success("[4] At place approach")

    logger.info("[5a] Move down to place pose")
    logger.debug(f"    place pose = {place_pose}")
    robot.set_cartesian_pose(place_pose, speed=CART_SPEED, acceleration=CART_ACCEL)
    logger.success("[5a] At place pose")

    logger.info("[5b] Open gripper to release")
    gripper.move(position=GRIPPER_OPEN_WIDTH_MM, force=GRIPPER_FORCE_N)
    logger.success("[5b] Released")

    logger.info("[5c] Retreat upward from place pose")
    robot.set_cartesian_pose(approach, speed=CART_SPEED, acceleration=CART_ACCEL)
    logger.success("[5c] Retreated from place")


def run_pipeline(robot, gripper):
    """Run the full pick-and-place pipeline across the grids."""
    pick_poses = generate_pick_poses()
    place_poses = generate_place_poses()

    n_parts = min(len(pick_poses), len(place_poses))
    logger.info(f"Starting pick-and-place pipeline for {n_parts} parts")
    robot.set_joint_positions(
        HOME_JOINT_POSE_DEG,
        speed=JOINT_SPEED_DEG,
        acceleration=JOINT_ACCEL_DEG,
    )
    for i in range(n_parts):
        logger.info(f"===== Part {i + 1} / {n_parts} =====")
        pick_part(robot, gripper, pick_poses[i])
        place_part(robot, gripper, place_poses[i])
        logger.success(f"===== Part {i + 1} complete =====")

    logger.info("[6] Move to home/safe pose")
    logger.debug(f"    home q = {HOME_JOINT_POSE_DEG}")
    robot.set_joint_positions(
        HOME_JOINT_POSE_DEG,
        speed=JOINT_SPEED_DEG,
        acceleration=JOINT_ACCEL_DEG,
    )
    logger.success("[6] At home pose — pipeline complete")


# ============================================================
# Main
# ============================================================

def main():
    robot = UniversalRobotsUR10E()
    gripper = OnRobotRG6()

    robot_connected = False
    gripper_connected = False

    try:
        logger.info(f"Connecting to UR10e at {ROBOT_IP}")
        robot.connect(ROBOT_IP)
        robot_connected = True
        logger.success("Robot connected")

        logger.info(f"Connecting to OnRobot RG6 at {GRIPPER_IP} via {GRIPPER_PROTOCOL}")
        gripper.connect(ip=GRIPPER_IP, protocol=GRIPPER_PROTOCOL)
        gripper_connected = True
        logger.success("Gripper connected")

        run_pipeline(robot, gripper)

    except Exception as e:
        logger.exception(f"Pipeline failed: {type(e).__name__}: {e}")
    finally:
        if gripper_connected:
            try:
                logger.info("Disconnecting gripper")
                gripper.disconnect()
            except Exception as e:
                logger.warning(f"Gripper disconnect failed: {e}")
        if robot_connected:
            try:
                logger.info("Disconnecting robot")
                robot.disconnect()
            except Exception as e:
                logger.warning(f"Robot disconnect failed: {e}")
        logger.info("Cleanup complete")


if __name__ == "__main__":
    main()
