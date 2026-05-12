"""
Robotiq 2F-85 connect and disconnect example for the Synapse SDK.

Currently illustrated with the Robotiq 2F-85; works the same with the
OnRobot RG6.

Usage:
    python connect_and_disconnect.py --ip <ROBOT_IP>
"""

import argparse
import time
from loguru import logger

from telekinesis.synapse.tools.parallel_grippers import robotiq


def main(ip: str):
    """Connect to a Robotiq 2F-85 at `ip` and cleanly disconnect."""

    # Create the gripper
    gripper = robotiq.Robotiq2F85()
    logger.info(f"Connecting Robotiq at {ip}...")

    # Connect to the gripper, 
    gripper.connect(ip=ip)
    logger.success("Connected.")

    # Sleep for a bit
    time.sleep(2.0)

    # Disconnect cleanly
    gripper.disconnect()
    logger.success("Disconnected.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Robotiq gripper connect/disconnect example")
    parser.add_argument("--ip", type=str, required=True, help="UR robot IP address")
    args = parser.parse_args()

    main(ip=args.ip)
