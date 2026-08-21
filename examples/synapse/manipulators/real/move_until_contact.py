"""
Drives the TCP down in -Z until contact is detected, then stops and reports the result.

Supports Universal Robots (UR), Epson, and virtual/sim.

Usage:
    python move_until_contact.py [--ip <ROBOT_IP>]
"""

import argparse

from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots


def main(ip: str) -> None:
    """Move the TCP down in -Z until contact is detected, then report and disconnect."""

    #===================== Create Robot ==========================================
    robot = universal_robots.UniversalRobotsUR10E(name='UR10e')

    # ==================== Visualization (Optional) =============================
    # Live: subscribes to the robot's state topic and redraws as it moves.
    robot.visualize_rerun(live=True)

    try:
        #===================== Connect Robot ==========================================
        robot.connect(ip=ip)

        # ==================== Run Skill ============================================
        contacted = robot.move_until_contact(
            cartesian_velocity=[0, 0, -0.02, 0, 0, 0],
            direction=[0, 0, 0, 0, 0, 0],
            acceleration=0.1,
        )
        logger.info(f"Contact detected: {contacted}")
    except (ConnectionError, OSError) as e:
        logger.error(f"Error occurred: {e}")
    finally:
        robot.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Move the TCP down until contact is detected")
    parser.add_argument("--ip", type=str, default="192.168.1.100", help="UR robot IP address (default: 192.168.1.100)")
    args = parser.parse_args()

    main(ip=args.ip)
