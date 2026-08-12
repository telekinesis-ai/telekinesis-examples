"""
Set Joint Positions (asynchronous) example for the Synapse SDK.

Commands an asynchronous joint move (joint 0 by +5 deg), then interrupts it
mid-trajectory with ``stop_joint_motion``.

Currently supported only for real hardware from Universal Robots

Usage:
    python set_joint_positions_async.py [--ip <ROBOT_IP>]
"""

import argparse
import time

from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots


def main(ip: str):
    """Run an asynchronous joint move and interrupt it mid-trajectory."""

    robot = universal_robots.UniversalRobotsUR5()
    robot.connect(ip=ip)

    try:
        # Asynchronous move: rotate joint 0 by +5 deg (returns immediately)
        target_joint_positions = robot.get_joint_positions()[:]
        target_joint_positions[0] += 5
        robot.set_joint_positions(
            joint_positions=target_joint_positions,
            speed=20,
            acceleration=20,
            asynchronous=True,
        )

        # Let it run briefly, then interrupt it mid-trajectory
        time.sleep(0.5)
        robot.stop_joint_motion(stopping_speed=20)
        logger.info(f"Stopped joint motion before reaching target joint positions: {target_joint_positions}")

    finally:
        # Disconnect
        robot.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Command an async joint move and interrupt it mid-trajectory")
    parser.add_argument("--ip", type=str, default="192.168.1.100", help="UR robot IP address (default: 192.168.1.100)")
    args = parser.parse_args()

    main(ip=args.ip)
