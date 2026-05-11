"""
I/O states examples for the Synapse SDK.

These examples run against real robot hardware. Currently supported only
for Universal Robots (UR10e).

Usage:
    python get_io_states.py --list
    python get_io_states.py --ip <ROBOT_IP> --example <NAME>
    python get_io_states.py --ip <ROBOT_IP> --all

Use --list to print the names of available examples without connecting to a
robot, so you can choose one to pass to --example. --ip is not required in
this mode because no hardware is contacted.
"""

import argparse
import difflib

from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots


def get_digital_in_state_example(ip: str):
    """Read state of digital input 0."""

    # Create and connect to the robot
    robot = universal_robots.UniversalRobotsUR10E()
    robot.connect(ip=ip)

    # Read digital input 0 and report the result
    try:
        logger.success(f"DI 0 state: {robot.get_digital_in_state(input_id=0)}")

    # Ensure we stop the robot and disconnect even if there was an error
    finally:
        robot.disconnect()


def get_digital_out_state_example(ip: str):
    """Read state of digital output 0."""

    # Create and connect to the robot
    robot = universal_robots.UniversalRobotsUR10E()
    robot.connect(ip=ip)

    # Read digital output 0 and report the result
    try:
        logger.success(f"DO 0 state: {robot.get_digital_out_state(output_id=0)}")

    # Ensure we stop the robot and disconnect even if there was an error
    finally:
        robot.disconnect()


def get_actual_digital_input_bits_example(ip: str):
    """Read all digital inputs as a bitmask."""

    # Create and connect to the robot
    robot = universal_robots.UniversalRobotsUR10E()
    robot.connect(ip=ip)

    # Read digital input bitmask and report the result
    try:
        logger.success(f"DI bits: {robot.get_actual_digital_input_bits():#018b}")

    # Ensure we stop the robot and disconnect even if there was an error
    finally:
        robot.disconnect()


def get_actual_digital_output_bits_example(ip: str):
    """Read all digital outputs as a bitmask."""

    # Create and connect to the robot
    robot = universal_robots.UniversalRobotsUR10E()
    robot.connect(ip=ip)

    # Read digital output bitmask and report the result
    try:
        logger.success(f"DO bits: {robot.get_actual_digital_output_bits():#018b}")

    # Ensure we stop the robot and disconnect even if there was an error
    finally:
        robot.disconnect()


def get_standard_analog_input_example(ip: str):
    """Read standard analog inputs 0 and 1."""

    # Create and connect to the robot
    robot = universal_robots.UniversalRobotsUR10E()
    robot.connect(ip=ip)

    # Read analog inputs 0 and 1 and report the results
    try:
        logger.success(f"AI 0: {robot.get_standard_analog_input(index=0):.4f}")
        logger.success(f"AI 1: {robot.get_standard_analog_input(index=1):.4f}")

    # Ensure we stop the robot and disconnect even if there was an error
    finally:
        robot.disconnect()


def get_standard_analog_output_example(ip: str):
    """Read standard analog outputs 0 and 1."""

    # Create and connect to the robot
    robot = universal_robots.UniversalRobotsUR10E()
    robot.connect(ip=ip)

    # Read analog outputs 0 and 1 and report the results
    try:
        logger.success(f"AO 0: {robot.get_standard_analog_output(index=0):.4f}")
        logger.success(f"AO 1: {robot.get_standard_analog_output(index=1):.4f}")

    # Ensure we stop the robot and disconnect even if there was an error
    finally:
        robot.disconnect()


def get_output_int_register_example(ip: str):
    """Read output integer register 18."""

    # Create and connect to the robot
    robot = universal_robots.UniversalRobotsUR10E()
    robot.connect(ip=ip)

    # Read output integer register 18 and report the result
    try:
        logger.success(f"Output int reg 18: {robot.get_output_int_register(output_id=18)}")

    # Ensure we stop the robot and disconnect even if there was an error
    finally:
        robot.disconnect()


def get_output_double_register_example(ip: str):
    """Read output double register 18."""

    # Create and connect to the robot
    robot = universal_robots.UniversalRobotsUR10E()
    robot.connect(ip=ip)

    # Read output double register 18 and report the result
    try:
        logger.success(f"Output double reg 18: {robot.get_output_double_register(output_id=18):.6f}")

    # Ensure we stop the robot and disconnect even if there was an error
    finally:
        robot.disconnect()


def get_example_dict(ip: str):
    return {
        "get_digital_in_state": lambda: get_digital_in_state_example(ip),
        "get_digital_out_state": lambda: get_digital_out_state_example(ip),
        "get_actual_digital_input_bits": lambda: get_actual_digital_input_bits_example(ip),
        "get_actual_digital_output_bits": lambda: get_actual_digital_output_bits_example(ip),
        "get_standard_analog_input": lambda: get_standard_analog_input_example(ip),
        "get_standard_analog_output": lambda: get_standard_analog_output_example(ip),
        "get_output_int_register": lambda: get_output_int_register_example(ip),
        "get_output_double_register": lambda: get_output_double_register_example(ip),
    }


def main():
    """
    Run an I/O states Synapse example.
    Usage:
        python get_io_states.py --list
        python get_io_states.py --ip <ROBOT_IP> --example <NAME>
        python get_io_states.py --ip <ROBOT_IP> --all

    Use --list to print the names of available examples without connecting to
    a robot, so you can choose one to pass to --example. --ip is not required
    in this mode because no hardware is contacted.
    """

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="I/O states Synapse examples")
    parser.add_argument("--ip", type=str, help="UR robot IP address")
    parser.add_argument("--example", type=str)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    # Handle example selection
    if args.list:
        for name in sorted(get_example_dict(ip="")):
            logger.info(f"  - {name}")
        return

    if not args.ip:
        parser.error("--ip is required unless --list is used.")
    examples = get_example_dict(ip=args.ip)
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
