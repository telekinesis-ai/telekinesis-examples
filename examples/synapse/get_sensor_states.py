"""
Sensor states examples for the Synapse SDK.

These examples run against real robot hardware. Currently supported only
for Universal Robots (UR10e).

Usage:
    python get_sensor_states.py --list
    python get_sensor_states.py --ip <ROBOT_IP> --example <NAME>
    python get_sensor_states.py --ip <ROBOT_IP> --all

Use --list to print the names of available examples without connecting to a
robot, so you can choose one to pass to --example. --ip is not required in
this mode because no hardware is contacted.
"""

import argparse
import difflib

from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots


def get_actual_tool_accelerometer_example(ip: str):
    """Read tool-mounted accelerometer reading [m/s²]."""

    # Create and connect to the robot
    robot = universal_robots.UniversalRobotsUR10E()
    robot.connect(ip=ip)

    # Read tool accelerometer and report the result
    try:
        logger.success(f"Tool accelerometer: {robot.get_actual_tool_accelerometer()}")

    # Ensure we stop the robot and disconnect even if there was an error
    finally:
        robot.disconnect()


def get_example_dict(ip: str):
    return {
        "get_actual_tool_accelerometer": lambda: get_actual_tool_accelerometer_example(ip),
    }


def main():
    """
    Run a Sensor states Synapse example.
    Usage:
        python get_sensor_states.py --list
        python get_sensor_states.py --ip <ROBOT_IP> --example <NAME>
        python get_sensor_states.py --ip <ROBOT_IP> --all

    Use --list to print the names of available examples without connecting to
    a robot, so you can choose one to pass to --example. --ip is not required
    in this mode because no hardware is contacted.
    """

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Sensor states Synapse examples")
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
