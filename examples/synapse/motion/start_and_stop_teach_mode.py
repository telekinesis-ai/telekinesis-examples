"""
Teach mode + manual waypoint capture example for the Synapse SDK.

There is no built-in "save waypoint" feature in teach mode — but combining
``start_teach_mode`` (zero-gravity back-drive) with ``get_cartesian_pose``
gives the standard teach-and-repeat pattern: hand-guide the arm, press
Enter to bookmark the current TCP pose, Ctrl-C to finish.

Currently supported only for Universal Robots (UR10e).

Usage:
    python teach_and_record_waypoints.py --ip <ROBOT_IP>
"""

import argparse
from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots


def main(robot_ip: str):
    """Enter teach mode, capture TCP poses on each Enter press, exit on Ctrl-C."""

    # Create robot instance
    robot = universal_robots.UniversalRobotsUR10E()

    # Connect to the robot
    robot.connect(ip=robot_ip)

    # Enter teach mode (zero-gravity back-drive, all axes free)
    logger.info("Starting teach mode")
    robot.start_teach_mode()

    # Capture waypoints on demand
    waypoints: list[list[float]] = []
    logger.info("Hand-guide the arm. Press Enter to capture a waypoint, Ctrl-C to finish.")
    try:
        while True:
            input()
            waypoints.append(robot.get_cartesian_pose())
            logger.success(f"Saved waypoint {len(waypoints)}: {waypoints[-1]}")
    except KeyboardInterrupt:
        logger.info(f"Capture finished — {len(waypoints)} waypoint(s) recorded.")

    # Exit teach mode and disconnect
    robot.stop_teach_mode()
    logger.success("Teach mode stopped.")
    robot.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="UR robot teach mode + manual waypoint capture example"
    )
    parser.add_argument("--ip", type=str, required=True, help="IP address of the UR robot")
    args = parser.parse_args()

    main(args.ip)
