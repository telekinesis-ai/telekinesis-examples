"""
Set Cartesian Pose (asynchronous) example for the Synapse SDK.

Commands an asynchronous 20 cm move up in Z, then interrupts it
mid-trajectory with ``stop_cartesian_motion``.

Currently supported only for real hardware from Universal Robots

Usage:
    python set_cartesian_pose_async.py [--ip <ROBOT_IP>]
"""

import argparse
import time

from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots


def main(ip: str):
    """Run an asynchronous Cartesian move and interrupt it mid-trajectory."""

    # Create robot instance
    robot = universal_robots.UniversalRobotsUR10E()
    robot.connect(ip=ip)

    # Create robot instance
    robot = universal_robots.UniversalRobotsUR10E()
    robot.connect(ip=args.ip)

    try:
        # Asynchronous move up 20 cm in Z (returns immediately)
        new_tcp_pose = robot.get_cartesian_pose()[:]
        new_tcp_pose[2] += 0.2
        robot.set_cartesian_pose(
            cartesian_pose=new_tcp_pose,
            speed=0.25,
            acceleration=0.25,
            asynchronous=True,
        )

        # Let it run briefly, then interrupt it mid-trajectory
        time.sleep(0.5)
        robot.stop_cartesian_motion(stopping_speed=0.25)
        logger.info(f"Stopped Cartesian motion before reaching target Cartesian pose: {new_tcp_pose}")

    finally:
        # Disconnect
        robot.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tool contact polling Synapse example")
    parser.add_argument("--ip", type=str, default="192.168.1.100", help="UR robot IP address (default: 192.168.1.100)")
    args = parser.parse_args()

    main(ip=args.ip)

