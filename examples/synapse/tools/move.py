"""
Robotiq 2F-85 move-to-position example for the Synapse SDK.

Configures the gripper to mm units, sets the stroke range to the 2F-85 max
(85 mm), then commands a synchronous move to 20 mm at 100% speed and 50%
force.

Usage:
    python move.py --ip <ROBOT_IP>
"""

import argparse
from loguru import logger

from telekinesis.synapse.tools.parallel_grippers import robotiq


def main(ip: str):
    """Move the gripper to 20 mm at 100% speed and 50% force."""

    # Create and connect to the gripper
    gripper = robotiq.Robotiq2F85()
    gripper.connect(ip=ip)

    try:
        # Configure the position unit and stroke range
        gripper.set_unit(parameter='position', unit='mm')
        gripper.set_position_range_mm(position_range_mm=85.0)

        # Command the move and report the resulting status and position
        status = gripper.move(
            position=20.0,
            speed=100.0,
            force=50.0,
            asynchronous=False,
        )
        logger.success(f"move() status: {status}, position: {gripper.get_current_position():.2f}")
    finally:
        gripper.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Robotiq gripper move example")
    parser.add_argument("--ip", type=str, required=True, help="UR robot IP address")
    args = parser.parse_args()

    main(ip=args.ip)
