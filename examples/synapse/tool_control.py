"""
Tools examples for the Synapse SDK.

These examples run against real robot hardware. 
Currently illustrated with the Robotiq 2F-85 gripper, but works with also
OnRobot RG6. More tools will be added in the future.

Usage:
    python tools_examples.py --ip <ROBOT_IP> --list
    python tools_examples.py --ip <ROBOT_IP> --example <NAME>
    python tools_examples.py --ip <ROBOT_IP> --all
"""

import argparse
import difflib

from loguru import logger

from telekinesis.synapse.tools.parallel_grippers import robotiq


def gripper_connect_disconnect_example(ip: str):
    """Connect to a Robotiq 2F-85 and cleanly disconnect."""

    # Create the gripper and connect, then disconnect cleanly
    gripper = robotiq.Robotiq2F85()
    logger.info(f"Connecting Robotiq at {ip}...")
    gripper.connect(ip=ip)
    logger.success("Connected.")
    gripper.disconnect()
    logger.success("Disconnected.")


def gripper_set_unit_example(ip: str):
    """Switch position unit to normalized (0..1)."""

    # Create and connect to the gripper
    gripper = robotiq.Robotiq2F85()
    gripper.connect(ip=ip)

    # Switch the gripper position unit to normalized and report success
    try:
        gripper.set_unit(parameter='position', unit='normalized')
        logger.success("Position unit set to 'normalized'.")

    # Ensure we disconnect even if there was an error
    finally:
        gripper.disconnect()


def gripper_set_position_range_example(ip: str):
    """Set the gripper stroke range to 85 mm (2F-85 max)."""

    # Create and connect to the gripper
    gripper = robotiq.Robotiq2F85()
    gripper.connect(ip=ip)

    # Set the gripper stroke range and report success
    try:
        gripper.set_position_range_mm(position_range_mm=85.0)
        logger.success("Position range set to 85 mm.")

    # Ensure we disconnect even if there was an error
    finally:
        gripper.disconnect()


def gripper_set_speed_example(ip: str):
    """Set default speed to 50% of max."""

    # Create and connect to the gripper
    gripper = robotiq.Robotiq2F85()
    gripper.connect(ip=ip)

    # Set the default gripper speed and report the effective value
    try:
        actual = gripper.set_speed(speed=50.0)
        logger.success(f"Default speed set; effective: {actual}")

    # Ensure we disconnect even if there was an error
    finally:
        gripper.disconnect()


def gripper_set_force_example(ip: str):
    """Set default grip force to 50%."""

    # Create and connect to the gripper
    gripper = robotiq.Robotiq2F85()
    gripper.connect(ip=ip)

    # Set the default gripper force and report the effective value
    try:
        actual = gripper.set_force(force=50.0)
        logger.success(f"Default force set; effective: {actual}")

    # Ensure we disconnect even if there was an error
    finally:
        gripper.disconnect()


def gripper_open_example(ip: str):
    """Open the gripper fully at 100% speed and 50% force."""

    # Create and connect to the gripper
    gripper = robotiq.Robotiq2F85()
    gripper.connect(ip=ip)

    # Open the gripper fully and report the resulting status and position
    try:
        status = gripper.open(speed=100.0, force=50.0, asynchronous=False)
        logger.success(f"open() status: {status}, position: {gripper.get_current_position():.2f}")

    # Ensure we disconnect even if there was an error
    finally:
        gripper.disconnect()


def gripper_close_example(ip: str):
    """Close the gripper fully at 100% speed and 50% force."""

    # Create and connect to the gripper
    gripper = robotiq.Robotiq2F85()
    gripper.connect(ip=ip)

    # Close the gripper fully and report the resulting status and position
    try:
        status = gripper.close(speed=100.0, force=50.0, asynchronous=False)
        logger.success(f"close() status: {status}, position: {gripper.get_current_position():.2f}")

    # Ensure we disconnect even if there was an error
    finally:
        gripper.disconnect()


def gripper_move_example(ip: str):
    """Move the gripper to 20 mm at 100% speed and 50% force."""

    # Create and connect to the gripper
    gripper = robotiq.Robotiq2F85()
    gripper.connect(ip=ip)

    # Configure the gripper to mm units and stroke range, then move to 20 mm
    try:

        # Configure the position unit and stroke range
        gripper.set_unit(parameter='position', unit='mm')
        gripper.set_position_range_mm(position_range_mm=85.0)

        # Command the move and report the resulting status and position
        status = gripper.move(position=20.0,
                              speed=100.0,
                              force=50.0,
                              asynchronous=False)
        logger.success(f"move() status: {status}, position: {gripper.get_current_position():.2f}")

    # Ensure we disconnect even if there was an error
    finally:
        gripper.disconnect()


def gripper_get_current_position_example(ip: str):
    """Read the gripper's current position in the configured unit."""

    # Create and connect to the gripper
    gripper = robotiq.Robotiq2F85()
    gripper.connect(ip=ip)

    # Read the current gripper position and report the result
    try:
        logger.success(f"Current position: {gripper.get_current_position()}")

    # Ensure we disconnect even if there was an error
    finally:
        gripper.disconnect()


def get_example_dict(ip: str):
    return {
        "gripper_connect_disconnect": lambda: gripper_connect_disconnect_example(ip),
        "gripper_set_unit": lambda: gripper_set_unit_example(ip),
        "gripper_set_position_range": lambda: gripper_set_position_range_example(ip),
        "gripper_set_speed": lambda: gripper_set_speed_example(ip),
        "gripper_set_force": lambda: gripper_set_force_example(ip),
        "gripper_open": lambda: gripper_open_example(ip),
        "gripper_close": lambda: gripper_close_example(ip),
        "gripper_move": lambda: gripper_move_example(ip),
        "gripper_get_current_position": lambda: gripper_get_current_position_example(ip),
    }


def main():
    """
    Run a Tools Synapse example.
    Usage:
        python tools_examples.py --ip <ROBOT_IP> --list
        python tools_examples.py --ip <ROBOT_IP> --example <NAME>
        python tools_examples.py --ip <ROBOT_IP> --all
    """

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Tools Synapse examples")
    parser.add_argument("--ip", type=str, required=True, help="UR robot IP address")
    parser.add_argument("--example", type=str)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()
    examples = get_example_dict(ip=args.ip)

    # Handle example selection
    if args.list:
        for name in sorted(examples):
            logger.info(f"  - {name}")
        return
    if args.all:
        for name, fn in examples.items():
            logger.info(f"Running {name}...")
            try:
                fn()
            except Exception as e:
                logger.error(f"{name} FAILED: {type(e).__name__}: {e}")
        return

    # Handle single example execution
    if not args.example:
        logger.error("Provide --example, --list, or --all.")
        raise SystemExit(1)
    name = args.example.lower()
    if name not in examples:
        matches = difflib.get_close_matches(name, examples.keys(), n=3, cutoff=0.4)
        logger.error(f"Example '{name}' not found.")
        if matches:
            logger.error("Did you mean: " + ", ".join(matches))
        raise SystemExit(1)
    examples[name]()


if __name__ == "__main__":
    main()
