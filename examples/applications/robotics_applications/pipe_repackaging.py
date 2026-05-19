# Pipeline: Pick aluminum cylindrical parts from an angled tray and drop them into a destination box.
# Hardware: UR10E robot at 192.168.2.2 + OnRobot RG2 gripper at 192.168.1.1.
# Strategy:
#   - Define all tunable constants at the top.
#   - Tray is tilted ~23 deg around base X. Compute pick poses by stepping in the tray-local frame
#     (tray-local X and Y), then transforming into base frame using the tray rotation.
#   - For each pick: pre-pick approach along tray normal, move_until_contact to find part,
#     close gripper, retreat, move to drop cell in destination box, open gripper, retreat.
#   - Log every step.

import numpy as np
from loguru import logger

from telekinesis.synapse import utils
from telekinesis.synapse.robots.manipulators import universal_robots
from telekinesis.synapse.tools.parallel_grippers import onrobot

# =====================================================================================
# TUNABLE CONSTANTS
# =====================================================================================

# --- Robot / gripper connection ---
ROBOT_IP = "192.168.2.2"
GRIPPER_IP = "192.168.1.1"
GRIPPER_PROTOCOL = "MODBUS_TCP"

# --- TCP offset (23 cm along tool flange Z) ---
TCP_OFFSET = [0.0, 0.0, 0.23, 0.0, 0.0, 0.0]  # [x, y, z, rx, ry, rz] (m, deg)

# --- Tray geometry ---
# Starting pick pose (top-left of the tray grid) in base frame, Euler degrees.
# The -23 deg Y rotation in this pose encodes the tray tilt orientation that
# the TCP must keep aligned with the tray surface for every pick.
TRAY_START_POSE = [-0.3869, 1.3208, 0.17429, -180.0, -23.0, 90.0]

# Tray is tilted around base X-axis by this angle (deg).
TRAY_TILT_DEG_X = 23.0

# Grid layout on the tray (rows step along tray-local Y, cols along tray-local X).
TRAY_ROWS = 2
TRAY_COLS = 3
TRAY_PITCH_X = 0.16  # spacing along tray-local X [m]
TRAY_PITCH_Y = -0.16  # spacing along tray-local Y [m]

# Approach / retreat along tray-local +Z (i.e. tray normal, away from surface).
TRAY_APPROACH_DIST = 0.08   # pre-pick standoff above the part [m]
TRAY_CONTACT_TRAVEL = 0.05  # max travel of move_until_contact past pre-pick [m]
TRAY_RETREAT_DIST = 0.03   # retreat distance after grasp [m]

# move_until_contact parameters: speed in TCP frame (linear m/s, angular deg/s).
# Move down along tray-local -Z. We push 'cartesian_speed' as a base-frame TCP velocity
# computed from the tray normal at runtime.
CONTACT_SPEED_LINEAR = 0.02  # m/s
CONTACT_ACC = 0.5            # m/s^2

# --- Motion speeds ---
MOVE_SPEED = 0.5            # TCP linear speed [m/s] for moveL
MOVE_ACC = 0.5              # TCP linear acc [m/s^2]

# --- Destination box (flat, not tilted) ---
DROP_START_POSE = [0.01, 0.74661, 0.07552, 180, 0, 90]  # top-left of drop grid (base frame, deg)
DROP_ROWS = 1
DROP_COLS = 6
DROP_PITCH_X = 0.062         # spacing along base X [m]
DROP_PITCH_Y = 0.062         # spacing along base Y [m]
DROP_APPROACH_DIST = 0.20   # pre-drop standoff above box cell [m]
DROP_RETREAT_DIST = 0.20    # retreat distance after release [m]

# --- Gripper ---
GRIPPER_GRASP_FORCE = 40.0  # N (RG2 max ~40 N)

# --- Home pose to return to at the end ---
HOME_JOINTS_DEG = [0.0, -90.0, -90.0, -90.0, 90.0, 0.0]
HOME_JOINT_SPEED = 30.0     # deg/s
HOME_JOINT_ACC = 60.0       # deg/s^2


# =====================================================================================
# HELPERS
# =====================================================================================

def tray_local_to_base(local_xyz: np.ndarray) -> np.ndarray:
    """
    Transform a 3D offset expressed in tray-local frame into a 3D offset in the base frame.
    The tray is rotated about the base X axis by TRAY_TILT_DEG_X.
    """
    a = np.deg2rad(TRAY_TILT_DEG_X)
    Rx = np.array([
        [1.0, 0.0, 0.0],
        [0.0, np.cos(a), -np.sin(a)],
        [0.0, np.sin(a), np.cos(a)],
    ])
    return Rx @ np.asarray(local_xyz, dtype=float)


def compute_tray_pick_pose(row: int, col: int) -> list[float]:
    """
    Compute the base-frame pick pose for tray cell (row, col).
    Orientation is kept identical to TRAY_START_POSE to remain aligned with tray surface.
    Translation is start_xyz + R_x(tray_tilt) * [col*pitch_x, row*pitch_y, 0].
    """
    start = np.array(TRAY_START_POSE, dtype=float)
    local_offset = np.array([col * TRAY_PITCH_X, row * TRAY_PITCH_Y, 0.0])
    base_offset = tray_local_to_base(local_offset)
    pose = start.copy()
    pose[0:3] = start[0:3] + base_offset
    return pose.tolist()


def offset_along_tray_normal(pose: list[float], distance: float) -> list[float]:
    """
    Offset a pose along the tray-local +Z axis (tray normal) by `distance` meters,
    expressed in the base frame.
    """
    base_offset = tray_local_to_base(np.array([0.0, 0.0, distance]))
    out = list(pose)
    out[0] += float(base_offset[0])
    out[1] += float(base_offset[1])
    out[2] += float(base_offset[2])
    return out


def compute_drop_pose(row: int, col: int) -> list[float]:
    """
    Compute base-frame drop pose for destination cell (row, col). Box is flat.
    """
    pose = list(DROP_START_POSE)
    pose[0] = DROP_START_POSE[0] + col * DROP_PITCH_X
    pose[1] = DROP_START_POSE[1] + row * DROP_PITCH_Y
    return pose


def offset_along_base_z(pose: list[float], distance: float) -> list[float]:
    """Offset a flat-box pose straight up along base +Z by `distance` meters."""
    out = list(pose)
    out[2] += float(distance)
    return out


def contact_speed_along_tray_normal_down() -> list[float]:
    """
    TCP velocity vector for move_until_contact, expressed in base frame.
    Pointing along tray-local -Z (into the tray surface). Angular components zero.
    """
    base_vel = tray_local_to_base(np.array([0.0, 0.0, -CONTACT_SPEED_LINEAR]))
    return [float(base_vel[0]), float(base_vel[1]), float(base_vel[2]), 0.0, 0.0, 0.0]


def contact_direction_along_tray_normal_down() -> list[float]:
    """
    Direction vector (unit, in base frame) used by move_until_contact to detect contacts.
    Same direction as the contact velocity but unit length; angular part zeroed.
    """
    base_dir = tray_local_to_base(np.array([0.0, 0.0, -1.0]))
    return [float(base_dir[0]), float(base_dir[1]), float(base_dir[2]), 0.0, 0.0, 0.0]


# =====================================================================================
# MAIN PIPELINE
# =====================================================================================

def main() -> None:
    robot = universal_robots.UniversalRobotsUR10E()
    gripper = onrobot.OnRobotRG2()

    robot_connected = False
    gripper_connected = False

    try:
        # ---- Step 1: Connect to robot ----
        logger.info(f"Connecting to UR10E at {ROBOT_IP} ...")
        robot.connect(ip=ROBOT_IP)
        robot_connected = True
        logger.info("UR10E connected.")

        # Apply TCP offset (23 cm along flange Z)
        logger.info(f"Setting TCP offset {TCP_OFFSET} on the controller.")
        robot.set_tcp(TCP_OFFSET)

        # ---- Step 2: Connect to gripper ----
        logger.info(f"Connecting to OnRobot RG2 at {GRIPPER_IP} (protocol={GRIPPER_PROTOCOL}) ...")
        gripper.connect(ip=GRIPPER_IP, protocol=GRIPPER_PROTOCOL)
        gripper_connected = True
        logger.info("OnRobot RG2 connected.")

        # ---- Step 3: Initialize gripper (open) ----
        logger.info("Opening gripper to initialize.")
        status = gripper.open(force=GRIPPER_GRASP_FORCE, asynchronous=False)
        logger.info(f"Gripper open status: {status}")

        # Pre-compute the contact velocity / direction used at every pick.
        contact_vel = contact_speed_along_tray_normal_down()
        contact_dir = contact_direction_along_tray_normal_down()
        logger.info(f"Tray-normal contact velocity (base frame): {contact_vel}")
        logger.info(f"Tray-normal contact direction (base frame): {contact_dir}")

        # ---- Step 4: Iterate over tray grid ----
        total_cells = TRAY_ROWS * TRAY_COLS
        max_drop_cells = DROP_ROWS * DROP_COLS
        n_to_pick = min(total_cells, max_drop_cells)
        logger.info(
            f"Tray grid: {TRAY_ROWS}x{TRAY_COLS} ({total_cells} cells). "
            f"Drop grid: {DROP_ROWS}x{DROP_COLS} ({max_drop_cells} cells). "
            f"Will process {n_to_pick} parts."
        )

        idx = 0
        for r in range(TRAY_ROWS):
            for c in range(TRAY_COLS):
                if idx >= n_to_pick:
                    break

                logger.info(f"--- Pick #{idx + 1}/{n_to_pick} :: tray cell (row={r}, col={c}) ---")

                # 4a: Pick pose at the tray surface (orientation aligned with tray).
                pick_pose = compute_tray_pick_pose(r, c)
                logger.info(f"Computed pick pose (base frame, deg): {pick_pose}")

                # 4b: Pre-pick approach above the tray surface along tray normal.
                pre_pick_pose = offset_along_tray_normal(pick_pose, TRAY_APPROACH_DIST)
                logger.info(f"Moving to pre-pick pose: {pre_pick_pose}")
                robot.set_cartesian_pose(
                    cartesian_pose=pre_pick_pose,
                    speed=MOVE_SPEED,
                    acceleration=MOVE_ACC,
                    asynchronous=False,
                )

                # 4c: move_until_contact downward along the tray normal to find the part.
                # logger.info(
                #     "Executing move_until_contact along tray normal "
                #     f"(speed={contact_vel}, direction={contact_dir})."
                # )
                # robot.move_until_contact(
                #     cartesian_speed=contact_vel,
                #     direction=contact_dir,
                #     acceleration=CONTACT_ACC,
                # )
                # logger.info("Contact detected on tray surface / part.")
                robot.set_cartesian_pose(
                    cartesian_pose=pick_pose,
                    speed=MOVE_SPEED,
                    acceleration=MOVE_ACC,
                    asynchronous=False,
                )

                # 4d: Close gripper to grasp.
                logger.info("Closing gripper to grasp the part.")
                grasp_status = gripper.close(force=GRIPPER_GRASP_FORCE, asynchronous=False)
                logger.info(f"Gripper close status: {grasp_status}")

                # 4e: Retreat along tray normal back to (above) pre-pick.
                retreat_pose = offset_along_tray_normal(pick_pose, TRAY_RETREAT_DIST)
                logger.info(f"Retreating along tray normal to: {retreat_pose}")
                robot.set_cartesian_pose(
                    cartesian_pose=retreat_pose,
                    speed=MOVE_SPEED,
                    acceleration=MOVE_ACC,
                    asynchronous=False,
                )

                # 4f: Compute drop pose for the corresponding destination cell.
                drop_r = idx // DROP_COLS
                drop_c = idx % DROP_COLS
                drop_pose = compute_drop_pose(drop_r, drop_c)
                logger.info(
                    f"Computed drop pose for box cell (row={drop_r}, col={drop_c}): {drop_pose}"
                )

                # 4g: Pre-drop above the drop cell.
                pre_drop_pose = offset_along_base_z(drop_pose, DROP_APPROACH_DIST)
                logger.info(f"Moving to pre-drop pose: {pre_drop_pose}")
                robot.set_cartesian_pose(
                    cartesian_pose=pre_drop_pose,
                    speed=MOVE_SPEED,
                    acceleration=MOVE_ACC,
                    asynchronous=False,
                )

                # 4h: Move down to drop pose.
                logger.info(f"Moving to drop pose: {drop_pose}")
                robot.set_cartesian_pose(
                    cartesian_pose=drop_pose,
                    speed=MOVE_SPEED,
                    acceleration=MOVE_ACC,
                    asynchronous=False,
                )

                # 4i: Open gripper to release / drop the part.
                logger.info("Opening gripper to release the part.")
                release_status = gripper.open(force=GRIPPER_GRASP_FORCE, asynchronous=False)
                logger.info(f"Gripper open status: {release_status}")

                # 4j: Retreat upward in base frame.
                post_drop_pose = offset_along_base_z(drop_pose, DROP_RETREAT_DIST)
                logger.info(f"Retreating from drop pose to: {post_drop_pose}")
                robot.set_cartesian_pose(
                    cartesian_pose=post_drop_pose,
                    speed=MOVE_SPEED,
                    acceleration=MOVE_ACC,
                    asynchronous=False,
                )

                logger.info(f"Pick #{idx + 1} complete.")
                idx += 1

            if idx >= n_to_pick:
                break

        # ---- Step 4 done; return home ----
        logger.info(f"All {idx} parts processed. Returning home: {HOME_JOINTS_DEG}")
        # robot.set_joint_positions(
        #     joint_positions=HOME_JOINTS_DEG,
        #     speed=HOME_JOINT_SPEED,
        #     acceleration=HOME_JOINT_ACC,
        #     asynchronous=False,
        # )
        logger.info("Robot at home position. Pipeline complete.")

    except Exception as e:
        logger.exception(f"Pipeline aborted due to error: {e}")
        # Best-effort safety stop on the robot if it is still connected.
        try:
            if robot_connected:
                logger.warning("Attempting to stop any ongoing robot motion.")
                robot.stop_cartesian_motion(stopping_speed=0.5)
        except Exception as stop_err:
            logger.error(f"Failed to stop robot motion cleanly: {stop_err}")
        raise

    finally:
        # ---- Hardware cleanup (always run) ----
        # Open gripper before disconnect so a part isn't left clamped.
        if gripper_connected:
            try:
                logger.info("Opening gripper before disconnect (safety).")
                gripper.open(force=GRIPPER_GRASP_FORCE, asynchronous=False)
            except Exception as g_err:
                logger.error(f"Failed to open gripper during cleanup: {g_err}")
            try:
                logger.info("Disconnecting OnRobot RG2.")
                gripper.disconnect()
            except Exception as g_err:
                logger.error(f"Error disconnecting gripper: {g_err}")

        if robot_connected:
            try:
                logger.info("Disconnecting UR10E.")
                robot.disconnect()
            except Exception as r_err:
                logger.error(f"Error disconnecting robot: {r_err}")

        logger.info("Cleanup complete.")


if __name__ == "__main__":
    main()
