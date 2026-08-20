"""
Read safety status bits example for the Synapse SDK.

``get_safety_status_bits`` returns the safety status as a bitmask
(bits 0-10: normal mode, reduced mode, protective stopped, recovery mode,
safeguard stopped, system e-stop, robot e-stop, e-stop, violation, fault,
stopped due to safety).

Currently supported only for real hardware, and only Universal Robots (UR).

Usage:
    python get_safety_status_bits.py [--ip <ROBOT_IP>]
"""

import argparse

from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots


def main(ip: str):
    """Log the raw safety status bitmask."""

    #===================== Create Robot ==========================================
    robot = universal_robots.UniversalRobotsUR10E()
    robot.connect(ip=ip)

    # ==================== Run Skill ============================================
    try:
        logger.success(f"safety_status_bits: {robot.get_safety_status_bits():#013b}")
    finally:
        robot.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Read safety status bits Synapse example")
    parser.add_argument("--ip", type=str, default="192.168.1.100", help="UR robot IP address (default: 192.168.1.100)")
    args = parser.parse_args()

    main(ip=args.ip)
