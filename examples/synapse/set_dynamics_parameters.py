"""
Dynamics setter examples for the Synapse SDK.

These examples run against real robot hardware. Currently supported only
for Universal Robots (UR10e).

Usage:
    python set_dynamics.py --list
    python set_dynamics.py --ip <ROBOT_IP> --example <NAME>
    python set_dynamics.py --ip <ROBOT_IP> --all

Use --list to print the names of available examples without connecting to a
robot, so you can choose one to pass to --example. --ip is not required in
this mode because no hardware is contacted.
"""

import argparse
import difflib

from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots


def set_gravity_example(ip: str):
    """Set gravity to standard upright mounting (Z=9.82)."""

    # Create and connect to the robot
    robot = universal_robots.UniversalRobotsUR10E()
    robot.connect(ip=ip)

    # Set gravity direction and report success
    try:
        robot.set_gravity(direction=[0.0, 0.0, 9.82])
        logger.success("Gravity set to [0.0, 0.0, 9.82] m/s².")

    # Ensure we stop the robot and disconnect even if there was an error
    finally:
        robot.disconnect()


def set_target_payload_example(ip: str):
    """Set target payload with mass, CoG, and full inertia tensor."""

    # Create and connect to the robot
    robot = universal_robots.UniversalRobotsUR10E()
    robot.connect(ip=ip)

    # Set target payload with mass, CoG, and inertia, then report success
    try:
        robot.set_target_payload(mass=1.5,
                                 cog=[0.0, 0.0, 0.05],
                                 inertia=[0.001, 0.001, 0.0005, 0.0, 0.0, 0.0])
        logger.success("Target payload set to 1.5 kg with CoG [0.0, 0.0, 0.05] m and inertia [0.001, 0.001, 0.0005, 0.0, 0.0, 0.0] kg·m².")

    # Ensure we stop the robot and disconnect even if there was an error
    finally:
        robot.disconnect()


def get_example_dict(ip: str):
    return {
        "set_gravity": lambda: set_gravity_example(ip),
        "set_target_payload": lambda: set_target_payload_example(ip),
    }


def main():
    """
    Run a Dynamics setter Synapse example.
    Usage:
        python set_dynamics.py --list
        python set_dynamics.py --ip <ROBOT_IP> --example <NAME>
        python set_dynamics.py --ip <ROBOT_IP> --all

    Use --list to print the names of available examples without connecting to
    a robot, so you can choose one to pass to --example. --ip is not required
    in this mode because no hardware is contacted.
    """

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Dynamics setter Synapse examples")
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
