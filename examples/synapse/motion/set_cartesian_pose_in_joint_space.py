"""
Set Cartesian Pose in Joint Space example for the Synapse SDK.

Moves to a target Cartesian pose using a trajectory that is linear in joint
space (joint-space ``moveJ`` with internal IK).

Real hardware is currently supported only for Universal Robots (UR10e).
Offline mode is supported for all manipulator brands — to run offline, omit
the ``robot.connect()`` / ``robot.disconnect()`` calls below; the SDK will
update the commanded-state cache (via IK) without touching hardware.

Usage:
    python set_cartesian_pose_in_joint_space.py --ip <ROBOT_IP>
"""

import argparse

from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots


def main(robot_ip: str):
    """Move the TCP to a target Cartesian pose via joint-space motion."""

    # Create robot instance
    robot = universal_robots.UniversalRobotsUR10E()

    # Target Cartesian pose [x, y, z (m), rx, ry, rz (deg)]
    target_pose = [-0.25, 0.25, 0.93, 90, 0, -104]

    # Safety warning before commanding real motion
    logger.warning(
        f"About to move real robot to target pose {target_pose}. "
        "Make sure it's safe to move there, otherwise use the advanced example."
    )

    # Connect to the robot
    robot.connect(ip=robot_ip)

    try:
        robot.set_cartesian_pose_in_joint_space(
            cartesian_pose=target_pose,
            speed=20,
            acceleration=20,
        )
        logger.info(f"Moved to target Cartesian pose: {target_pose}")
    finally:
        robot.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="UR10e set_cartesian_pose_in_joint_space example"
    )
    parser.add_argument("--ip", type=str, required=True, help="IP address of the UR robot")
    args = parser.parse_args()

    main(args.ip)
