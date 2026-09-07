"""
Logs a normalized Robot controller status snapshot.

Supports Epson and Universal Robots (UR).

Usage:
    python get_status.py [--ip <ROBOT_IP>]
"""

import argparse

from loguru import logger

from telekinesis.synapse.robots.manipulators import epson


def main(ip: str | None) -> None:
    """Log the current normalized Epson controller status."""

    #===================== Create Robot ==========================================
    robot = epson.EpsonCX4A601S(name='EpsonCX4A601S')

    try:
        #===================== Connect Robot ==========================================
        if ip:
            robot.connect(ip=ip)

        # ==================== Run Skill ============================================
        status = robot.get_status()
        logger.success(f"status: {status}")
    except (ConnectionError, OSError) as e:
        logger.error(f"Error occurred: {e}")
    finally:
        robot.disconnect()
        robot.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Read normalized Epson status Synapse example")
    parser.add_argument("--ip", type=str, default=None,
                         help="Epson controller IP address for real hardware, e.g. 192.168.0.1")
    args = parser.parse_args()

    main(ip=args.ip)
