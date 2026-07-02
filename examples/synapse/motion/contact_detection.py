"""
Contact Detection example for the Synapse SDK.

Drives the TCP slowly downward (-Z) while polling contact detection, and stops
as soon as the tool touches a surface. This exercises the lower-level contact
API (start/read/stop); for the single-call helper see move_until_contact.py.

Demonstrates:
- `start_contact_detection()`
- `read_contact_detection()`
- `stop_contact_detection()`

Currently supported only for real hardware from Universal Robots

Usage:
    python contact_detection.py [--ip <ROBOT_IP>]
"""

import argparse
import time

from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots


def main(ip: str):
    """Probe downward until contact is detected, then stop and report."""

    robot = universal_robots.UniversalRobotsUR10E()
    robot.connect(ip=ip)

    try:
        # Move the TCP down by 15 cm, asynchronously, while polling for contact.
        target_pose = robot.get_cartesian_pose()
        target_pose[2] -= 0.15

        robot.start_contact_detection()
        robot.set_cartesian_pose(cartesian_pose=target_pose, speed=0.05,
                                 acceleration=0.25, asynchronous=True)

        # Poll until contact, or for up to 5 s if nothing is hit.
        contact = False
        deadline = time.time() + 5.0
        while not contact and time.time() < deadline:
            contact = robot.read_contact_detection()
            time.sleep(0.02)

        robot.stop_cartesian_motion(stopping_speed=0.25)
        robot.stop_contact_detection()
        logger.success(f"Contact: {contact}")

    finally:
        robot.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tool contact polling Synapse example")
    parser.add_argument("--ip", type=str, default="192.168.1.100", help="UR robot IP address (default: 192.168.1.100)")
    args = parser.parse_args()

    main(ip=args.ip)

