"""
Enters teach mode, captures TCP poses on each Enter press, and exits on Ctrl-C.

Supports Universal Robots (UR).

Usage:
    python start_and_stop_teach_mode.py [--ip <ROBOT_IP>]
"""

import argparse

from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots


def main(ip: str) -> None:
    """Enter teach mode, capture TCP poses on each Enter press, exit on Ctrl-C."""

    #===================== Create Robot ==========================================
    robot = universal_robots.UniversalRobotsUR10E(name='UR10e')

    # ==================== Visualization (Optional) =============================
    robot.visualize_rerun(live=True)

    try:
        #===================== Connect Robot ==========================================
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
    except (ConnectionError, OSError) as e:
        logger.error(f"Error occurred: {e}")
    finally:
        robot.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Teach mode with manual TCP waypoint capture")
    parser.add_argument("--ip", type=str, default="192.168.1.100", help="UR robot IP address (default: 192.168.1.100)")
    args = parser.parse_args()

    main(ip=args.ip)
