"""
Set Cartesian Pose in joint space (relative) example for the Synapse SDK.

Moves the TCP to a target Cartesian pose with a trajectory linear in joint
space (as opposed to ``set_cartesian_pose``, which is linear in Cartesian
space). On real hardware this dispatches to the UR ``move_j_ik`` RTDE call.

Currently supported only for real hardware. Works on Universal Robots (UR) and Epson.

Usage:
    python set_cartesian_pose_in_joint_space.py [--ip <ROBOT_IP>]
"""

import argparse

from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots


def main(ip: str):
    """Move the TCP to a target pose along a joint-space-linear trajectory."""

    #===================== Create Robot ==========================================
    robot = universal_robots.UniversalRobotsUR10E(name='UR10e')
    robot.connect(ip=ip)

    # ==================== Visualization (Optional) =============================
    # Live: subscribes to the robot's state topic and redraws as it moves.
    robot.visualize_rerun()

    # ==================== Run Skill ============================================
    try:
        # Define a target relative to the current pose
        current_cartesian_pose = robot.get_cartesian_pose()
        target_cartesian_pose = current_cartesian_pose.copy()
        target_cartesian_pose[2] += 0.1  # Move 10 cm up in Z

        robot.set_cartesian_pose_in_joint_space(
            cartesian_pose=target_cartesian_pose,
            speed=60,
            acceleration=80,
        )
        logger.info(f"Moved to target Cartesian pose: {target_cartesian_pose}")
    finally:
        robot.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Move the TCP to a target pose along a joint-space trajectory")
    parser.add_argument("--ip", type=str, default="192.168.1.100", help="UR robot IP address (default: 192.168.1.100)")
    args = parser.parse_args()

    main(ip=args.ip)
