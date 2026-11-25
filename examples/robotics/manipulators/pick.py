"""
Picks one object with a Robotiq Hand-E attached to a UR10e, and visualizes it in Rerun.

Supports Isaac Sim. The same calls run on real hardware once the robot and the
gripper are connected to it instead.

Usage:
    python pick.py --prim_path <ROBOT_PRIM_PATH> --gripper_prim_path <GRIPPER_PRIM_PATH>
    python pick.py --prim_path <ROBOT_PRIM_PATH> --gripper_prim_path <GRIPPER_PRIM_PATH> --save-rrd pick.rrd
"""

import argparse
import time

import numpy as np
import rerun as rr
from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots
from telekinesis.synapse.tools.parallel_grippers import robotiq
from telekinesis.tf import tfutils

# Cartesian pose [x, y, z, rx, ry, rz] in meters and Euler XYZ degrees, robot base frame.
PICK_POSE = [0.09, 0.509, 0.15, 180, 0, 0]
APPROACH_OFFSET_Z = 0.1  # m, pre-pick distance above the pick pose
HOME_JOINTS = [-90, -90, 90, -90, -90, 0]  # degrees

# Length of the axis arrows drawn at the pick poses, in meters.
AXIS_LENGTH = 0.05


def init_rerun(app_id: str, save_rrd: str | None) -> "rr.RecordingStream":
    """Start a Rerun recording, live in the viewer or saved to an .rrd file."""
    rr.init(app_id, spawn=save_rrd is None)
    recording = rr.get_global_data_recording()
    if save_rrd:
        recording.save(save_rrd)
    return recording


def log_frame(path: str, base_T_frame: np.ndarray) -> None:
    """Draw a 4 x 4 pose as three colored axes: x red, y green, z blue."""
    origin = base_T_frame[:3, 3]
    rr.log(path,
           rr.Arrows3D(origins=[origin, origin, origin],
                       vectors=[base_T_frame[:3, axis] * AXIS_LENGTH for axis in range(3)],
                       colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]]))


def main(prim_path: str, gripper_prim_path: str, save_rrd: str | None) -> None:
    """Pick one object and lift it."""

    #===================== Start Rerun ==========================================
    recording = init_rerun("telekinesis_synapse_pick", save_rrd)

    #===================== Create Robot and Gripper =============================
    robot = universal_robots.UniversalRobotsUR10E(name="ur10e")
    gripper = robotiq.RobotiqHandE()

    # ==================== Visualization (Optional) =============================
    # The robot follows the simulation live. Attaching the gripper below draws it
    # on the flange, and the pick pose and the pre-pick pose are drawn as axes.
    robot.visualize_rerun(recording_stream=recording)

    base_T_pick = tfutils.pose_to_transformation_matrix(PICK_POSE, rot_type="deg")
    pick_T_pre_pick = np.eye(4)
    pick_T_pre_pick[2, 3] = -APPROACH_OFFSET_Z
    log_frame("/pick/pick_pose", base_T_pick)
    log_frame("/pick/pre_pick_pose", base_T_pick @ pick_T_pre_pick)

    try:
        #===================== Connect Robot and Gripper ========================
        robot.connect(simulation_prim_path=prim_path)
        gripper.connect(simulation_prim_path=gripper_prim_path)
        robot.attach_tool(gripper)

        # ==================== Run Skill ========================================
        robot.set_joint_positions(HOME_JOINTS)
        robot.pick(PICK_POSE, APPROACH_OFFSET_Z)
        logger.success("Picked the object.")
        rr.log("/pick/status", rr.TextLog("Picked the object."))

        # Give the live view time to draw the final pose before the recording ends.
        time.sleep(1.0)
    except (ConnectionError, OSError, RuntimeError) as e:
        logger.error(f"Error occurred: {e}")
        rr.log("/pick/status", rr.TextLog(f"Pick failed: {e}"))
    finally:
        if gripper.is_connected:
            gripper.disconnect()
        robot.disconnect()
        robot.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pick Synapse example")
    parser.add_argument("--prim_path", type=str, required=True,
                         help='Isaac Sim robot prim path, e.g. '
                              '"/World/vention_table/robot_mount/UR10e_assembly/UR10e"')
    parser.add_argument("--gripper_prim_path", type=str, required=True,
                         help='Isaac Sim gripper prim path, e.g. '
                              '"/World/vention_table/robot_mount/UR10e_assembly/Hand_E"')
    parser.add_argument("--save-rrd", type=str, default=None,
                         help="Save the Rerun recording to this .rrd file instead of opening the viewer")
    args = parser.parse_args()

    main(prim_path=args.prim_path,
         gripper_prim_path=args.gripper_prim_path,
         save_rrd=args.save_rrd)
