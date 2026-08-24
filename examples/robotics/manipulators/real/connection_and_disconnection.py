"""
Connects to a UR10e, waits briefly, then cleanly disconnects.

Supports Universal Robots (UR) and Epson.

Usage:
    python connection_and_disconnection.py [--ip <ROBOT_IP>]
"""

import argparse
import time

from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots


def main(ip: str) -> None:
    """Connect to a UR10e at `ip` and cleanly disconnect."""

    #===================== Create Robot ==========================================
    robot = universal_robots.UniversalRobotsUR10E(name='UR10e')

    try:
        # ==================== Run Skill ============================================
        robot.connect(ip=ip)
        logger.success(f"Connected to UR10e at {ip}.")
        
        time.sleep(2)
    except (ConnectionError, OSError) as e:
        logger.error(f"Error occurred: {e}")
    finally:
        robot.disconnect()
        logger.success("Disconnected.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Connection Synapse example")
    parser.add_argument("--ip", type=str, default="192.168.1.100", help="UR robot IP address (default: 192.168.1.100)")
    args = parser.parse_args()

    main(ip=args.ip)
