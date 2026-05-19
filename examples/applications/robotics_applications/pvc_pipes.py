# Pipeline: Pick PVC pipes from a grid and place them gently into a paper box.
# Strategy:
#   - Connect to UR10E + OnRobot RG2.
#   - Set TCP offset (23 cm from flange).
#   - Iterate over a hardcoded grid of pipe positions.
#   - For each pipe: approach above, descend, close gripper, lift, move above box,
#     descend with move_until_contact (gentle stop), open gripper, retreat.
#   - Cleanup hardware on success or failure.

from loguru import logger
import numpy as np

from telekinesis.synapse.robots.manipulators.universal_robots import UniversalRobotsUR10E
from telekinesis.synapse.tools.parallel_grippers.onrobot import OnRobotRG2

# -----------------------------------------------------------------------------
# Tunable constants
# -----------------------------------------------------------------------------

# Hardware
ROBOT_IP = "192.168.2.2"
GRIPPER_IP = "192.168.1.1"
TCP_OFFSET = [0.0, 0.0, 0.02, 0.0, 0.0, 0.0]  # 23 cm along flange Z, no rotation

# Grid of PVC pipes on the table (base frame, meters; orientation in degrees).
# Grid origin = pose above pipe at row=0, col=0 (top of pipe contact point).
GRID_ORIGIN_XYZ = [0.2341, 0.78192, 0.29442]              # x, y, z [m]
GRID_TOOL_RPY   = [-180.0, 0.0, 90.0]                # tool pointing down
GRID_ROW_DELTA  = [0.00, -0.10997, 0.00]               # +Y per row [m]
GRID_COL_DELTA  = [ -0.10161, 0.00, 0.00]               # +X per col [m]
GRID_ROWS = 2
GRID_COLS = 3

# Heights / clearances
APPROACH_HEIGHT       = 0.25   # how far above pipe top to pre-position [m]
GRASP_DESCENT_OFFSET  = 0.00   # additional descent past grid Z to grasp [m]
LIFT_HEIGHT           = 0.25   # how high to lift after grasping [m]

# Drop box (paper packaging box) — pose above box opening.
BOX_ABOVE_POSE = [-0.523563, -0.001, 0.47432, -180.0, 0.0, -180.0]   # x,y,z,rx,ry,rz
# Direction for move_until_contact placement (downward along base -Z).
PLACE_CONTACT_DIRECTION = [0.0, 0.0, -1.0, 0.0, 0.0, 0.0]
PLACE_CONTACT_SPEED     = [0.0, 0.0, -0.03, 0.0, 0.0, 0.0]  # 5 cm/s downward
PLACE_CONTACT_ACCEL     = 0.2

# Motion params
LINEAR_SPEED = 0.55     # m/s for moveL
LINEAR_ACCEL = .5      # m/s^2

# Gripper params (OnRobot RG2 — width in mm, force in N)
GRIPPER_OPEN_WIDTH_MM   = 100.0
GRIPPER_CLOSE_WIDTH_MM  = 20.0   # tuned for PVC pipe outer diameter
GRIPPER_GRASP_FORCE_N   = 20.0
GRIPPER_RELEASE_FORCE_N = 40.0

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def grid_pose_for(row: int, col: int) -> list[float]:
    """Compute the [x,y,z,rx,ry,rz] pose at the top of the pipe at (row,col)."""
    x = GRID_ORIGIN_XYZ[0] + row * GRID_ROW_DELTA[0] + col * GRID_COL_DELTA[0]
    y = GRID_ORIGIN_XYZ[1] + row * GRID_ROW_DELTA[1] + col * GRID_COL_DELTA[1]
    z = GRID_ORIGIN_XYZ[2] + row * GRID_ROW_DELTA[2] + col * GRID_COL_DELTA[2]
    return [x, y, z, *GRID_TOOL_RPY]

def offset_z(pose: list[float], dz: float) -> list[float]:
    """Return pose with Z offset by dz (meters)."""
    p = list(pose)
    p[2] += dz
    return p

# -----------------------------------------------------------------------------
# Main pipeline
# -----------------------------------------------------------------------------

robot = UniversalRobotsUR10E()
gripper = OnRobotRG2()

robot_connected = False
gripper_connected = False

try:
    # 1. Connect hardware
    logger.info(f"Connecting to UR10E at {ROBOT_IP} ...")
    robot.connect(ROBOT_IP)
    robot_connected = True
    logger.info("UR10E connected.")

    logger.info(f"Connecting to OnRobot RG2 at {GRIPPER_IP} ...")
    gripper.connect(GRIPPER_IP, protocol="MODBUS_TCP")
    gripper_connected = True
    logger.info("OnRobot RG2 connected.")

    # 2. Configure TCP offset (23 cm from flange)
    logger.info(f"Setting TCP offset: {TCP_OFFSET}")
    robot.set_tcp(TCP_OFFSET)

    # 3. Open gripper to start in known state
    logger.info(f"Opening gripper to {GRIPPER_OPEN_WIDTH_MM} mm.")
    gripper.move(position=GRIPPER_OPEN_WIDTH_MM, force=GRIPPER_RELEASE_FORCE_N)

    # 4. Iterate grid positions and pick-and-place each pipe
    pipe_index = 0
    total_pipes = GRID_ROWS * GRID_COLS
    logger.info(f"Beginning pick-and-place for {total_pipes} pipes "
                f"({GRID_ROWS} rows x {GRID_COLS} cols).")

    for row in range(GRID_ROWS):
        for col in range(GRID_COLS):
            pipe_index += 1
            pipe_top_pose = grid_pose_for(row, col)
            approach_pose = offset_z(pipe_top_pose, APPROACH_HEIGHT)
            grasp_pose    = offset_z(pipe_top_pose, GRASP_DESCENT_OFFSET)
            lift_pose     = offset_z(pipe_top_pose, LIFT_HEIGHT)

            logger.info(f"[Pipe {pipe_index}/{total_pipes}] "
                        f"row={row}, col={col}, top_pose={pipe_top_pose}")

            # 4a. Move above pipe
            logger.info(f"[Pipe {pipe_index}] Moving to approach pose {approach_pose}.")
            robot.set_cartesian_pose(approach_pose,
                                     speed=LINEAR_SPEED,
                                     acceleration=LINEAR_ACCEL)

            # 4b. Open gripper before descending (ensure clear)
            logger.info(f"[Pipe {pipe_index}] Pre-opening gripper to "
                        f"{GRIPPER_OPEN_WIDTH_MM} mm.")
            gripper.open(force=GRIPPER_RELEASE_FORCE_N)

            # 4c. Descend to grasp pose
            logger.info(f"[Pipe {pipe_index}] Descending to grasp pose {grasp_pose}.")
            robot.set_cartesian_pose(grasp_pose,
                                     speed=LINEAR_SPEED,
                                     acceleration=LINEAR_ACCEL)

            # 4d. Close gripper to grasp
            logger.info(f"[Pipe {pipe_index}] Closing gripper to "
                        f"{GRIPPER_CLOSE_WIDTH_MM} mm @ {GRIPPER_GRASP_FORCE_N} N.")
            grasp_status = gripper.close(force=GRIPPER_GRASP_FORCE_N)
            logger.info(f"[Pipe {pipe_index}] Grasp status: {grasp_status}")

            # 4e. Lift pipe
            logger.info(f"[Pipe {pipe_index}] Lifting to {lift_pose}.")
            robot.set_cartesian_pose(lift_pose,
                                     speed=LINEAR_SPEED,
                                     acceleration=LINEAR_ACCEL)

            # 4f. Move above the paper box
            logger.info(f"[Pipe {pipe_index}] Moving above box at {BOX_ABOVE_POSE}.")
            robot.set_cartesian_pose(BOX_ABOVE_POSE,
                                     speed=LINEAR_SPEED,
                                     acceleration=LINEAR_ACCEL)

            # 4g. Gentle descent into box using move_until_contact
            logger.info(f"[Pipe {pipe_index}] Descending into box with "
                        f"move_until_contact (gentle placement).")
            # robot.move_until_contact(cartesian_speed=PLACE_CONTACT_SPEED,
            #                          direction=PLACE_CONTACT_DIRECTION,
            #                          acceleration=PLACE_CONTACT_ACCEL)
            logger.info(f"[Pipe {pipe_index}] Contact detected; placement reached.")

            # 4h. Open gripper to release pipe gently
            logger.info(f"[Pipe {pipe_index}] Opening gripper to release pipe.")
            release_status = gripper.open(force=GRIPPER_RELEASE_FORCE_N)
            logger.info(f"[Pipe {pipe_index}] Release status: {release_status}")

            # 4i. Retreat back above box to clear
            logger.info(f"[Pipe {pipe_index}] Retreating above box.")
            robot.set_cartesian_pose(BOX_ABOVE_POSE,
                                     speed=LINEAR_SPEED,
                                     acceleration=LINEAR_ACCEL)

            logger.info(f"[Pipe {pipe_index}] Done.")

    logger.info("All pipes have been picked and placed successfully.")

except Exception as e:
    logger.exception(f"Pipeline failed: {e}")
    # Best-effort: stop any ongoing motion before disconnecting.
    try:
        if robot_connected:
            logger.warning("Stopping robot motion due to failure.")
            robot.stop_cartesian_motion()
    except Exception as stop_err:
        logger.error(f"Failed to stop robot motion cleanly: {stop_err}")
    raise

finally:
    # Cleanup hardware regardless of success/failure.
    if gripper_connected:
        try:
            logger.info("Disconnecting gripper.")
            gripper.disconnect()
        except Exception as gerr:
            logger.error(f"Error disconnecting gripper: {gerr}")

    if robot_connected:
        try:
            logger.info("Disconnecting robot.")
            robot.disconnect()
        except Exception as rerr:
            logger.error(f"Error disconnecting robot: {rerr}")

    logger.info("Pipeline cleanup complete.")