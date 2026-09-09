"""
Turns off the physical robot motors.

Supports Universal Robots (UR) and Epson.

Usage:
    python motor_off.py [--ip <ROBOT_IP>]
"""

import argparse

from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots


def main(ip: str | None) -> None:
    """Turn off physical robot motors."""

    #===================== Create Robot ==========================================
    robot = universal_robots.UniversalRobotsUR10E(name='UR10e')

    try:
        #===================== Connect Robot ==========================================
        if ip:
            robot.connect(ip=ip)

        # ==================== Run Skill ============================================
        robot.motor_off()
        logger.success("Robot motors turned off.")
    except (ConnectionError, OSError) as e:
        logger.error(f"Error occurred: {e}")
    finally:
        robot.disconnect()
        robot.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Turn off robot motors Synapse example")
    parser.add_argument("--ip", type=str, default=None,
                         help="UR robot IP address for real hardware, e.g. 192.168.1.100")
    args = parser.parse_args()

    main(ip=args.ip)
