"""
Set Joint Position in Cartesian Space example for the Synapse SDK.

Moves to a target joint configuration using a trajectory that is linear in
Cartesian space (Cartesian ``moveL`` derived from joint positions via FK).

Real hardware is currently supported only for Universal Robots (UR10e).
Offline mode is supported for all manipulator brands — to run offline, omit
the ``robot.connect()`` / ``robot.disconnect()`` calls below; the SDK will
update the commanded-state cache (via FK) without touching hardware.

Usage:
    python set_joint_position_in_cartesian_space.py --ip <ROBOT_IP>
"""

import argparse

from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots


def main(robot_ip: str):
    """Move to a target joint configuration via Cartesian-space motion."""

    # Create robot instance
    robot = universal_robots.UniversalRobotsUR10E()

    # Target joint positions in degrees
    q_target = [0, -90, 0, -90, 0, 0]

    # Safety warning before commanding real motion
    logger.warning(
        f"About to move real robot to joint positions {q_target}. "
        "Make sure it's safe to move there, otherwise use the advanced example."
    )

    # Connect to the robot
    robot.connect(ip=robot_ip)

    try:
        robot.set_joint_position_in_cartesian_space(
            joint_positions=q_target,
        )
        logger.info(f"Moved to target joint positions: {q_target}")
    finally:
        robot.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="UR10e set_joint_position_in_cartesian_space example"
    )
    parser.add_argument("--ip", type=str, required=True, help="IP address of the UR robot")
    args = parser.parse_args()

    main(args.ip)
