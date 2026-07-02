"""
Set Cartesian Pose in Joint Space (asynchronous) example for the Synapse SDK.

Commands an asynchronous Cartesian move (10 cm up in Z) using joint-space
motion, then interrupts it mid-trajectory with ``stop_joint_motion``.

Currently supported only for real hardware from Universal Robots

Usage:
    python set_cartesian_pose_in_joint_space_async.py [--ip <ROBOT_IP>]
"""

import argparse
import time

from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots


def main():
    """Run an asynchronous Cartesian-in-joint-space move and interrupt it mid-trajectory."""

    parser = argparse.ArgumentParser(
        description="UR robot asynchronous set_cartesian_pose_in_joint_space example"
    )
    parser.add_argument("--ip", type=str, default="192.168.1.100", help="IP address of the UR robot (default: 192.168.1.100)")
    args = parser.parse_args()

    # Create robot instance
    robot = universal_robots.UniversalRobotsUR10E()
    robot.connect(ip=args.ip)

    try:
        # Asynchronous move up 10 cm in Z using joint-space motion (returns immediately)
        target_pose = robot.get_cartesian_pose()[:]
        target_pose[2] += 0.1
        robot.set_cartesian_pose_in_joint_space(
            cartesian_pose=target_pose,
            speed=20,
            acceleration=40,
            asynchronous=True,
        )

        # Let it run briefly, then interrupt it mid-trajectory
        time.sleep(0.5)
        robot.stop_joint_motion(stopping_speed=20)
        logger.info(f"Stopped joint motion before reaching: {target_pose}")

    finally:
        # Disconnect
        robot.disconnect()


if __name__ == "__main__":
    main()
