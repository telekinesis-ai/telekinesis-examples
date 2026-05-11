"""
Controller states examples for the Synapse SDK.

These examples run against real robot hardware. Currently supported only
for Universal Robots (UR10e).

Usage:
    python get_controller_states.py --list
    python get_controller_states.py --ip <ROBOT_IP> --example <NAME>
    python get_controller_states.py --ip <ROBOT_IP> --all

Use --list to print the names of available examples without connecting to a
robot, so you can choose one to pass to --example. --ip is not required in
this mode because no hardware is contacted.
"""

import argparse
import difflib

from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots


def get_actual_execution_time_example(ip: str):
    """Read controller execution time [s]."""

    # Create and connect to the robot
    robot = universal_robots.UniversalRobotsUR10E()
    robot.connect(ip=ip)

    # Read controller execution time and report the result
    try:
        logger.success(f"Execution time: {robot.get_actual_execution_time()} s")

    # Ensure we stop the robot and disconnect even if there was an error
    finally:
        robot.disconnect()


def get_speed_scaling_combined_example(ip: str):
    """Read combined speed-scaling factor (program × safety)."""

    # Create and connect to the robot
    robot = universal_robots.UniversalRobotsUR10E()
    robot.connect(ip=ip)

    # Read combined speed-scaling factor and report the result
    try:
        logger.success(f"Speed scaling combined: {robot.get_speed_scaling_combined()}")

    # Ensure we stop the robot and disconnect even if there was an error
    finally:
        robot.disconnect()


def get_step_time_example(ip: str):
    """Read controller step time [s]."""

    # Create and connect to the robot
    robot = universal_robots.UniversalRobotsUR10E()
    robot.connect(ip=ip)

    # Read controller step time and report the result
    try:
        logger.success(f"Step time: {robot.get_step_time()} s")

    # Ensure we stop the robot and disconnect even if there was an error
    finally:
        robot.disconnect()


def get_speed_scaling_example(ip: str):
    """Read current speed scaling factor (0..1)."""

    # Create and connect to the robot
    robot = universal_robots.UniversalRobotsUR10E()
    robot.connect(ip=ip)

    # Read speed scaling factor and report the result
    try:
        logger.success(f"Speed scaling: {robot.get_speed_scaling()}")

    # Ensure we stop the robot and disconnect even if there was an error
    finally:
        robot.disconnect()


def get_async_operation_progress_example(ip: str):
    """Read async-operation progress string."""

    # Create and connect to the robot
    robot = universal_robots.UniversalRobotsUR10E()
    robot.connect(ip=ip)

    # Read async-operation progress and report the result
    try:
        logger.success(f"Async progress: {robot.get_async_operation_progress()}")

    # Ensure we stop the robot and disconnect even if there was an error
    finally:
        robot.disconnect()


def get_freedrive_status_example(ip: str):
    """Read freedrive-status integer."""

    # Create and connect to the robot
    robot = universal_robots.UniversalRobotsUR10E()
    robot.connect(ip=ip)

    # Read freedrive status and report the result
    try:
        logger.success(f"Freedrive status: {robot.get_freedrive_status()}")

    # Ensure we stop the robot and disconnect even if there was an error
    finally:
        robot.disconnect()


def get_example_dict(ip: str):
    return {
        "get_actual_execution_time": lambda: get_actual_execution_time_example(ip),
        "get_speed_scaling_combined": lambda: get_speed_scaling_combined_example(ip),
        "get_step_time": lambda: get_step_time_example(ip),
        "get_speed_scaling": lambda: get_speed_scaling_example(ip),
        "get_async_operation_progress": lambda: get_async_operation_progress_example(ip),
        "get_freedrive_status": lambda: get_freedrive_status_example(ip),
    }


def main():
    """
    Run a Controller states Synapse example.
    Usage:
        python get_controller_states.py --list
        python get_controller_states.py --ip <ROBOT_IP> --example <NAME>
        python get_controller_states.py --ip <ROBOT_IP> --all

    Use --list to print the names of available examples without connecting to
    a robot, so you can choose one to pass to --example. --ip is not required
    in this mode because no hardware is contacted.
    """

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Controller states Synapse examples")
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
