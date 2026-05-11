"""
Electrical examples for the Synapse SDK.

These examples run against real robot hardware. Currently supported only
for Universal Robots (UR10e).

Usage:
    python electrical_examples.py --ip <ROBOT_IP> --list
    python electrical_examples.py --ip <ROBOT_IP> --example <NAME>
    python electrical_examples.py --ip <ROBOT_IP> --all
"""

import argparse
import difflib

from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots


def get_actual_main_voltage_example(ip: str):
    """Read main voltage [V]."""

    # Create and connect to the robot
    robot = universal_robots.UniversalRobotsUR10E()
    robot.connect(ip=ip)

    # Read main voltage and report the result
    try:
        logger.success(f"Main voltage: {robot.get_actual_main_voltage()} V")

    # Ensure we stop the robot and disconnect even if there was an error
    finally:
        robot.disconnect()


def get_actual_robot_voltage_example(ip: str):
    """Read robot voltage [V]."""

    # Create and connect to the robot
    robot = universal_robots.UniversalRobotsUR10E()
    robot.connect(ip=ip)

    # Read robot voltage and report the result
    try:
        logger.success(f"Robot voltage: {robot.get_actual_robot_voltage()} V")

    # Ensure we stop the robot and disconnect even if there was an error
    finally:
        robot.disconnect()


def get_actual_robot_current_example(ip: str):
    """Read robot current [A]."""

    # Create and connect to the robot
    robot = universal_robots.UniversalRobotsUR10E()
    robot.connect(ip=ip)

    # Read robot current and report the result
    try:
        logger.success(f"Robot current: {robot.get_actual_robot_current()} A")

    # Ensure we stop the robot and disconnect even if there was an error
    finally:
        robot.disconnect()


def get_actual_joint_voltage_example(ip: str):
    """Read joint voltages [V]."""

    # Create and connect to the robot
    robot = universal_robots.UniversalRobotsUR10E()
    robot.connect(ip=ip)

    # Read joint voltages and report the result
    try:
        logger.success(f"Joint voltages: {robot.get_actual_joint_voltage()}")

    # Ensure we stop the robot and disconnect even if there was an error
    finally:
        robot.disconnect()


def get_actual_current_as_torque_example(ip: str):
    """Read joint currents converted to torques [N·m]."""

    # Create and connect to the robot
    robot = universal_robots.UniversalRobotsUR10E()
    robot.connect(ip=ip)

    # Read joint currents converted to torques and report the result
    try:
        logger.success(f"Current-as-torque: {robot.get_actual_current_as_torque()}")

    # Ensure we stop the robot and disconnect even if there was an error
    finally:
        robot.disconnect()


def get_example_dict(ip: str):
    return {
        "get_actual_main_voltage": lambda: get_actual_main_voltage_example(ip),
        "get_actual_robot_voltage": lambda: get_actual_robot_voltage_example(ip),
        "get_actual_robot_current": lambda: get_actual_robot_current_example(ip),
        "get_actual_joint_voltage": lambda: get_actual_joint_voltage_example(ip),
        "get_actual_current_as_torque": lambda: get_actual_current_as_torque_example(ip),
    }


def main():
    """
    Run an Electrical Synapse example.
    Usage:
        python electrical_examples.py --ip <ROBOT_IP> --list
        python electrical_examples.py --ip <ROBOT_IP> --example <NAME>
        python electrical_examples.py --ip <ROBOT_IP> --all
    """

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Electrical Synapse examples")
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
