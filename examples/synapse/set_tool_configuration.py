"""
Set Tool configuration examples for the Synapse SDK.

These examples run against real robot hardware. Currently supported only
for Universal Robots (UR10e).

Usage:
    python tool_configuration.py --list
    python tool_configuration.py --ip <ROBOT_IP> --example <NAME>
    python tool_configuration.py --ip <ROBOT_IP> --all

Use --list to print the names of available examples without connecting to a
robot, so you can choose one to pass to --example. --ip is not required in
this mode because no hardware is contacted.
"""

import argparse
import difflib

from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots


def set_payload_example(ip: str):
    """Set payload to 2 kg with CoG 5 cm along tool Z."""

    # Create and connect to the robot
    robot = universal_robots.UniversalRobotsUR10E()
    robot.connect(ip=ip)

    # Set payload mass and CoG, then read it back
    try:

        # Set the payload mass and center of gravity
        robot.set_payload(mass=2.0, cog=[0.0, 0.0, 0.05])

        # Read back the configured payload values
        logger.success(
            f"Payload set; readback: mass={robot.get_payload()}, cog={robot.get_payload_cog()}"
        )

    # Ensure we stop the robot and disconnect even if there was an error
    finally:
        robot.disconnect()


def set_tcp_example(ip: str):
    """Set TCP offset to 10 cm along flange Z."""

    # Create and connect to the robot
    robot = universal_robots.UniversalRobotsUR10E()
    robot.connect(ip=ip)

    # Set the TCP offset and read it back
    try:

        # Set the TCP offset
        tcp = [0.0, 0.0, 0.1, 0.0, 0.0, 0.0]
        robot.set_tcp(tcp_offset=tcp)

        # Read back the active TCP offset
        logger.success(f"TCP set; readback: {robot.get_tcp_offset()}")

    # Ensure we stop the robot and disconnect even if there was an error
    finally:
        robot.disconnect()


def get_example_dict(ip: str):
    return {
        "set_payload": lambda: set_payload_example(ip),
        "set_tcp": lambda: set_tcp_example(ip),
    }


def main():
    """
    Run a Tool configuration Synapse example.
    Usage:
        python tool_configuration.py --list
        python tool_configuration.py --ip <ROBOT_IP> --example <NAME>
        python tool_configuration.py --ip <ROBOT_IP> --all

    Use --list to print the names of available examples without connecting to
    a robot, so you can choose one to pass to --example. --ip is not required
    in this mode because no hardware is contacted.
    """

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Tool configuration Synapse examples")
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
