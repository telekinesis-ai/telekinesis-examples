"""
Safety state examples for the Synapse SDK.

These examples read safety-subsystem state and run pre-flight safety
validators (pose/joint within safety limits). Currently supported only
for Universal Robots (UR10e).

Usage:
    python get_safety_states.py --list
    python get_safety_states.py --ip <ROBOT_IP> --example <NAME>
    python get_safety_states.py --ip <ROBOT_IP> --all

Use --list to print the names of available examples without connecting to a
robot, so you can choose one to pass to --example. --ip is not required in
this mode because no hardware is contacted.
"""

import argparse
import difflib

from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots


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


def get_safety_status_bits_example(ip: str):
    """Read the safety-status bitmask."""

    # Create and connect to the robot
    robot = universal_robots.UniversalRobotsUR10E()
    robot.connect(ip=ip)

    # Read safety-status bitmask and report the result
    try:
        logger.success(f"Safety bits: {robot.get_safety_status_bits():#013b}")

    # Ensure we stop the robot and disconnect even if there was an error
    finally:
        robot.disconnect()


def is_protective_stopped_example(ip: str):
    """Check whether the robot is in a protective stop."""

    # Create and connect to the robot
    robot = universal_robots.UniversalRobotsUR10E()
    robot.connect(ip=ip)

    # Check protective-stop status and report the result
    try:
        logger.success(f"Is protective stopped: {robot.is_protective_stopped()}")

    # Ensure we stop the robot and disconnect even if there was an error
    finally:
        robot.disconnect()


def is_emergency_stopped_example(ip: str):
    """Check whether the robot is in an emergency stop."""

    # Create and connect to the robot
    robot = universal_robots.UniversalRobotsUR10E()
    robot.connect(ip=ip)

    # Check emergency-stop status and report the result
    try:
        logger.success(f"Is emergency stopped: {robot.is_emergency_stopped()}")

    # Ensure we stop the robot and disconnect even if there was an error
    finally:
        robot.disconnect()


def is_pose_within_safety_limits_example(ip: str):
    """Check whether a candidate pose is within the configured safety limits."""

    # Create and connect to the robot
    robot = universal_robots.UniversalRobotsUR10E()
    robot.connect(ip=ip)

    # Check whether a candidate pose is within safety limits and report the result
    try:
        pose = [0.5, 0.0, 0.5, 0.0, 0.0, 0.0]
        logger.success(f"Pose within limits: {robot.is_pose_within_safety_limits(pose=pose)}")

    # Ensure we stop the robot and disconnect even if there was an error
    finally:
        robot.disconnect()


def is_joints_within_safety_limits_example(ip: str):
    """Check whether a candidate joint configuration is within the configured safety limits."""

    # Create and connect to the robot
    robot = universal_robots.UniversalRobotsUR10E()
    robot.connect(ip=ip)

    # Check whether a candidate joint configuration is within safety limits and report the result
    try:
        q = [0.0, -90.0, -90.0, 0.0, 90.0, 0.0]
        logger.success(f"Joints within limits: {robot.is_joints_within_safety_limits(q=q)}")

    # Ensure we stop the robot and disconnect even if there was an error
    finally:
        robot.disconnect()


def get_example_dict(ip: str):
    return {
        "get_safety_mode": lambda: get_safety_mode_example(ip),
        "get_safety_status_bits": lambda: get_safety_status_bits_example(ip),
        "is_protective_stopped": lambda: is_protective_stopped_example(ip),
        "is_emergency_stopped": lambda: is_emergency_stopped_example(ip),
        "is_pose_within_safety_limits": lambda: is_pose_within_safety_limits_example(ip),
        "is_joints_within_safety_limits": lambda: is_joints_within_safety_limits_example(ip),
    }


def main():
    """
    Run a Safety state Synapse example.
    Usage:
        python get_safety_states.py --list
        python get_safety_states.py --ip <ROBOT_IP> --example <NAME>
        python get_safety_states.py --ip <ROBOT_IP> --all

    Use --list to print the names of available examples without connecting to
    a robot, so you can choose one to pass to --example. --ip is not required
    in this mode because no hardware is contacted.
    """

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Safety state Synapse examples")
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
