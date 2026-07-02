"""
Start and stop jog mode example for the Synapse SDK.

``start_jog`` drives the TCP continuously at a Cartesian twist
``[vx, vy, vz (m/s), ωx, ωy, ωz (deg/s)]`` expressed in the ``feature``
frame, until ``stop_jog`` is called — equivalent to holding a direction
button on the teach pendant. ``feature`` selects the frame:
``0`` = base, ``1`` = tool, ``2`` = custom (provide ``custom_frame``).
This is Cartesian jogging — there is no joint-jog API.

Currently supported only for real hardware from Universal Robots

Usage:
    python start_and_stop_jog_mode.py [--ip <ROBOT_IP>]
"""

import argparse
import time
from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots


def main(ip: str):
    """Jog the TCP -Z at 5 cm/s in the base frame for 5 seconds, then stop."""

    # Create and connect to the robot
    robot = universal_robots.UniversalRobotsUR10E()
    robot.connect(ip=ip)
    
    # Cartesian twist [vx, vy, vz (m/s), ωx, ωy, ωz (deg/s)] in the base frame
    cartesian_velocity = [0.0, 0.0, 0.05, 0.0, 0.0, 0.0]
    logger.info(f"Starting jog - cartesian_velocity [m/s, deg/s]: {cartesian_velocity}")
    robot.start_jog(
        cartesian_velocity=cartesian_velocity,
        feature=0,
        cartesian_acceleration=0.5,
    )

    # Let the jog run, then stop
    time.sleep(5.0)
    robot.stop_jog()
    logger.success("Jog mode stopped.")

    # Disconnect
    robot.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tool contact polling Synapse example")
    parser.add_argument("--ip", type=str, default="192.168.1.100", help="UR robot IP address (default: 192.168.1.100)")
    args = parser.parse_args()

    main(ip=args.ip)

