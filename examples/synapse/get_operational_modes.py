"""
Robot operational mode examples for the Synapse SDK.

These examples read the controller's high-level mode/status codes — robot
mode, robot status, safety mode, and runtime state. Currently supported
only for Universal Robots (UR10e).

Usage:
    python get_robot_operational_modes.py --list
    python get_robot_operational_modes.py --ip <ROBOT_IP> --example <NAME>
    python get_robot_operational_modes.py --ip <ROBOT_IP> --all

Use --list to print the names of available examples without connecting to a
robot, so you can choose one to pass to --example. --ip is not required in
this mode because no hardware is contacted.
"""

import argparse
import difflib

from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots


def get_robot_mode_example(ip: str):
    """Read robot mode string."""

    # Create and connect to the robot
    robot = universal_robots.UniversalRobotsUR10E()
    robot.connect(ip=ip)

    # Read robot mode and report the result
    try:
        logger.success(f"Robot mode: {robot.get_robot_mode()}")

    # Ensure we stop the robot and disconnect even if there was an error
    finally:
        robot.disconnect()


def get_robot_status_example(ip: str):
    """Read robot status string."""

    # Create and connect to the robot
    robot = universal_robots.UniversalRobotsUR10E()
    robot.connect(ip=ip)

    # Read robot status and report the result
    try:
        logger.success(f"Robot status: {robot.get_robot_status()}")

    # Ensure we stop the robot and disconnect even if there was an error
    finally:
        robot.disconnect()


def get_safety_mode_example(ip: str):
    """Read safety mode string."""

    # Create and connect to the robot
    robot = universal_robots.UniversalRobotsUR10E()
    robot.connect(ip=ip)

    # Read safety mode and report the result
    try:
        logger.success(f"Safety mode: {robot.get_safety_mode()}")

    # Ensure we stop the robot and disconnect even if there was an error
    finally:
        robot.disconnect()


def get_runtime_state_example(ip: str):
    """Read runtime state string."""

    # Create and connect to the robot
    robot = universal_robots.UniversalRobotsUR10E()
    robot.connect(ip=ip)

    # Read runtime state and report the result
    try:
        logger.success(f"Runtime state: {robot.get_runtime_state()}")

    # Ensure we stop the robot and disconnect even if there was an error
    finally:
        robot.disconnect()


def get_example_dict(ip: str):
    return {
        "get_robot_mode": lambda: get_robot_mode_example(ip),
        "get_robot_status": lambda: get_robot_status_example(ip),
        "get_safety_mode": lambda: get_safety_mode_example(ip),
        "get_runtime_state": lambda: get_runtime_state_example(ip),
    }


def main():
    """
    Run a Robot operational mode Synapse example.
    Usage:
        python get_robot_operational_modes.py --list
        python get_robot_operational_modes.py --ip <ROBOT_IP> --example <NAME>
        python get_robot_operational_modes.py --ip <ROBOT_IP> --all

    Use --list to print the names of available examples without connecting to
    a robot, so you can choose one to pass to --example. --ip is not required
    in this mode because no hardware is contacted.
    """

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Robot operational mode Synapse examples")
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
