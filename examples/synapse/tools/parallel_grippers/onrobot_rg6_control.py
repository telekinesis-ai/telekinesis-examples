"""
Demo for the OnRobot RG6 gripper wrapper.

Demonstrates:
- ``connect()``, ``open()``, ``close()``, and ``move()`` methods.

Usage:
    python onrobot_rg6_control.py --ip <ROBOT_IP>
"""

import argparse

from telekinesis.synapse.tools.parallel_grippers.onrobot import OnRobotRG6


def main(ip):
    """Run a gripper open-close demonstration using OnRobotRG6."""
    gripper = OnRobotRG6()
    gripper.connect(ip)
    try:
        print(f"Current position: {gripper.get_current_position()} mm")

        status = gripper.open()
        print(f"Open: {status}")

        status = gripper.close()
        print(f"Close: {status}")

        status = gripper.move(80.0)
        print(f"Move to 80 mm: {status}")

    finally:
        gripper.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Demo for the OnRobot RG6 gripper wrapper")
    parser.add_argument("--ip", type=str, default="192.168.1.100", help="OnRobot RG6 IP address (default: 192.168.1.100)")
    args = parser.parse_args()

    main(ip=args.ip)
