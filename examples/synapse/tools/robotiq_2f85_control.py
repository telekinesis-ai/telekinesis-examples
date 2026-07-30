"""Robotiq control example.

Demonstrates how to connect to a Robotiq gripper and use:
- open()
- move()
- close()

Usage example:
    python robotiq_2f85_control.py --ip <ROBOT_IP>
"""

import argparse
import time
from loguru import logger
from telekinesis.synapse.tools.parallel_grippers import robotiq


def main(ip: str, serial_port: str, protocol: str) -> None:
    """Run a minimal Robotiq connect/open/move/close example."""

    # Connection parameters
    timeout_ms = 5000
    verbose = True

    # Create a Robotiq gripper instance
    gripper = robotiq.Robotiq2F85()

    try:
        logger.info(f"Connecting to Robotiq gripper at {ip}...")

        # Connect
        gripper.connect(
            ip=ip,
            serial_port=serial_port,
            protocol=protocol,
            timeout_ms=timeout_ms,
            verbose=verbose,
        )

        # Configure gripper units
        # Robotiq supports mm for position values.
        # Robotiq defaults speed/force to percent units.

        position_range_mm = 40  # mm (2F-85 has a maximum stroke of 85mm)
        default_speed = (
            100.0  # % (for Robotiq, speed is interpreted as percentage of maximum speed)
        )
        default_force = 100.0  # % (for Robotiq, force is interpreted as grip strength percentage)

        # Set gripper units and parameters
        gripper.set_unit(parameter="position", unit="mm")
        gripper.set_position_range_mm(position_range_mm)
        gripper.set_speed(default_speed)
        gripper.set_force(default_force)

        # Open
        logger.info("Opening gripper...")
        open_status = gripper.open(
            speed=default_speed,
            force=default_force,
            asynchronous=False,
        )
        logger.info(f"open() status: {open_status}")
        current_position = gripper.get_current_position()
        logger.info(f"Current position after open(): {current_position:.2f} mm")

        time.sleep(2)  # Wait for 2 seconds before next command

        # Move
        move_status = gripper.move(
            position=30,
            speed=default_speed,
            force=default_force,
            asynchronous=False,
        )
        logger.info(f"move() status: {move_status}")
        current_position = gripper.get_current_position()
        logger.info(f"Current position after move(): {current_position:.2f} mm")

        # Close
        logger.info("Closing gripper...")
        close_status = gripper.close(
            speed=default_speed,
            force=default_force,
            asynchronous=False,
        )
        logger.info(f"close() status: {close_status}")
        current_position = gripper.get_current_position()
        logger.info(f"Current position after close(): {current_position:.2f} mm")

    finally:
        
        # Disconnect
        logger.info("Disconnecting gripper...")
        gripper.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Robotiq open/move/close example using Synapse.",
    )
    parser.add_argument(
        "--protocol",
        type=str,
        default="URCAP",
        choices=["URCAP", "MODBUS_RTU"],
        help="Transport: URCAP (UR controller) or MODBUS_RTU (direct USB).",
    )
    parser.add_argument(
        "--ip",
        type=str,
        default=None,
        help="Robot Controller IP address (used with protocol=URCAP).",
    )
    parser.add_argument(
        "--serial-port",
        dest="serial_port",
        type=str,
        default="COM4",
        help="Serial port string (used with protocol=MODBUS_RTU). "
        "e.g. COM4 (Windows), /dev/ttyUSB0 (Linux), "
        "/dev/cu.usbserial-XXXX (macOS).",
    )
    cli_args = parser.parse_args()

    main(ip=cli_args.ip, serial_port=cli_args.serial_port, protocol=cli_args.protocol)
