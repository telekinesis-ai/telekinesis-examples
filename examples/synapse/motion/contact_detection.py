"""
Contact Detection example for the Synapse SDK.

Shows the full lifecycle: ``start_contact_detection`` arms the watcher,
``read_contact_detection`` polls during motion, and ``stop_contact_detection``
disarms it and returns the final result.

Currently supported only for Universal Robots (UR10e).

Usage:
    python contact_detection.py --ip <ROBOT_IP>
"""

import argparse
import difflib
import time

from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots


def start_contact_detection_example(ip: str):
    """Start contact detection on a UR10e at `ip` during an async +Z move."""

    # Create and connect to the robot
    robot = universal_robots.UniversalRobotsUR10E()
    robot.connect(ip=ip)

    # Run an asynchronous move and watch for contact detection
    try:

        # Get current pose and define a target slightly above in Z
        current = robot.get_cartesian_pose()
        target = list(current)
        target[2] += 0.002

        # Start an asynchronous move and then start contact detection
        robot.set_cartesian_pose(cartesian_pose=target, 
                                 speed=0.05, 
                                 acceleration=0.1, 
                                 asynchronous=True)
        robot.start_contact_detection()

        # Watch for contact detection for up to 1 s, then stop and report the final result
        for _ in range(20):
            if robot.read_contact_detection():
                logger.info("Contact detected!")
                break
            time.sleep(0.05)

        # Stop the move and contact detection, and report the final result
        contact = robot.stop_contact_detection()
        logger.success(f"Final contact: {contact}")

    # Ensure we stop the robot and disconnect even if there was an error
    finally:
        robot.disconnect()


def read_contact_detection_example(ip: str):
    """Read whether contact has been detected on a UR10e at `ip`."""

    # Create and connect to the robot
    robot = universal_robots.UniversalRobotsUR10E()
    robot.connect(ip=ip)

    # Start contact detection and read the result
    try:
        robot.start_contact_detection()
        contact = robot.read_contact_detection()
        logger.success(f"Contact: {contact}")
        robot.stop_contact_detection()

    # Ensure we stop the robot and disconnect even if there was an error
    finally:
        robot.disconnect()


def stop_contact_detection_example(ip: str):
    """Stop contact detection on a UR10e at `ip` and return the final result."""

    # Create and connect to the robot
    robot = universal_robots.UniversalRobotsUR10E()
    robot.connect(ip=ip)

    # Start contact detection, wait a moment, then stop and report the result
    try:
        robot.start_contact_detection()
        result = robot.stop_contact_detection()
        logger.success(f"Stop result: {result}")

    # Ensure we stop the robot and disconnect even if there was an error
    finally:
        robot.disconnect()


def get_example_dict(ip: str):
    return {
        "start_contact_detection": lambda: start_contact_detection_example(ip),
        "read_contact_detection": lambda: read_contact_detection_example(ip),
        "stop_contact_detection": lambda: stop_contact_detection_example(ip),
    }


def main():
    """
    Run a Contact Detection Synapse example.
    Usage:
        python contact_detection_examples.py --list
        python contact_detection_examples.py --ip <ROBOT_IP> --example <NAME>
        python contact_detection_examples.py --ip <ROBOT_IP> --all

    Use --list to print the names of available examples without connecting to
    a robot, so you can choose one to pass to --example. --ip is not required
    in this mode because no hardware is contacted.
    """

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Contact Detection Synapse examples")
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
