"""
Triggers a protective stop on the controller, immediately halting all motion.

Supports Universal Robots (UR), Epson, and virtual/sim.

Usage:
    python trigger_protective_stop.py [--ip <ROBOT_IP>]
"""

import argparse

from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots


def main(ip: str) -> None:
    """Trigger a protective stop on the controller."""

    #===================== Create Robot ==========================================
    robot = universal_robots.UniversalRobotsUR10E(name='UR10e')

    try:
        #===================== Connect Robot ==========================================
        robot.connect(ip=ip)

        # ==================== Run Skill ============================================
        robot.trigger_protective_stop()
        logger.success("Protective stop triggered.")
    except (ConnectionError, OSError) as e:
        logger.error(f"Error occurred: {e}")
    finally:
        robot.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trigger a protective stop on the robot controller")
    parser.add_argument("--ip", type=str, default="192.168.1.100", help="UR robot IP address (default: 192.168.1.100)")
    args = parser.parse_args()

    main(ip=args.ip)
