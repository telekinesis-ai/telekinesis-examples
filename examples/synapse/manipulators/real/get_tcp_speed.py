"""
Read TCP speed example for the Synapse SDK.

Returns the current TCP velocity ``[vx, vy, vz, ωx, ωy, ωz]`` (m/s, deg/s).
Reads from ``self.state``, which the control loop keeps up to date from the
connected backend.

Currently supported only for real hardware. Works on Universal Robots (UR) and Epson.

Usage:
    python get_tcp_speed.py [--ip <ROBOT_IP>]
"""

import argparse

from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots


def main(ip: str):
    """Log the live TCP speed [m/s, deg/s]."""

    #===================== Create Robot ==========================================
    robot = universal_robots.UniversalRobotsUR10E()
    robot.connect(ip=ip)

    # ==================== Run Skill ============================================
    try:
        logger.success(f"tcp_speed [m/s, deg/s]: {robot.get_tcp_speed()}")
    finally:
        robot.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Read TCP speed Synapse example")
    parser.add_argument("--ip", type=str, default="192.168.1.100", help="UR robot IP address (default: 192.168.1.100)")
    args = parser.parse_args()

    main(ip=args.ip)
