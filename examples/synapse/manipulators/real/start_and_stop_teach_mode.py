"""
Teach mode + manual waypoint capture example for the Synapse SDK.

There is no built-in "save waypoint" feature in teach mode — but combining
``start_teach_mode`` (zero-gravity back-drive) with ``get_cartesian_pose``
gives the standard teach-and-repeat pattern: hand-guide the arm, press
Enter to bookmark the current TCP pose, Ctrl-C to finish.

Currently supported only for real hardware, and only Universal Robots (UR).

Usage:
    python start_and_stop_teach_mode.py [--ip <ROBOT_IP>]
"""

import argparse
from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots


def main(ip: str):
    """Enter teach mode, capture TCP poses on each Enter press, exit on Ctrl-C."""

    #===================== Create Robot ==========================================
    robot = universal_robots.UniversalRobotsUR10E()
    robot.connect(ip=ip)

    # ==================== Run Skill ============================================
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

    # Exit teach mode
    robot.stop_teach_mode()
    logger.success("Teach mode stopped.")

    # ==================== Visualization (Optional) =============================
    robot.visualize_rerun(live=False)

    # Disconnect
    robot.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Teach mode with manual TCP waypoint capture")
    parser.add_argument("--ip", type=str, default="192.168.1.100", help="UR robot IP address (default: 192.168.1.100)")
    args = parser.parse_args()

    main(ip=args.ip)

