"""
Dynamics getter examples for the Synapse SDK.

These examples run against real robot hardware. Currently supported only
for Universal Robots (UR10e).

Usage:
    python get_dynamics_states.py --list
    python get_dynamics_states.py --ip <ROBOT_IP> --example <NAME>
    python get_dynamics_states.py --ip <ROBOT_IP> --all

Use --list to print the names of available examples without connecting to a
robot, so you can choose one to pass to --example. --ip is not required in
this mode because no hardware is contacted.
"""

import argparse
import difflib

from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots


def get_mass_matrix_example(ip: str):
    """Read the joint-space mass matrix (6x6 flat)."""

    # Create and connect to the robot
    robot = universal_robots.UniversalRobotsUR10E()
    robot.connect(ip=ip)

    # Read the joint-space mass matrix and report the result
    try:
        logger.success(f"Mass matrix: {robot.get_mass_matrix()}")

    # Ensure we stop the robot and disconnect even if there was an error
    finally:
        robot.disconnect()


def get_coriolis_and_centrifugal_torques_example(ip: str):
    """Read Coriolis and centrifugal torques [N·m]."""

    # Create and connect to the robot
    robot = universal_robots.UniversalRobotsUR10E()
    robot.connect(ip=ip)

    # Read Coriolis and centrifugal torques and report the result
    try:
        logger.success(f"Coriolis/centrifugal torques: {robot.get_coriolis_and_centrifugal_torques()}")

    # Ensure we stop the robot and disconnect even if there was an error
    finally:
        robot.disconnect()


def get_actual_momentum_example(ip: str):
    """Read actual generalized momentum p = M(q)·q̇."""

    # Create and connect to the robot
    robot = universal_robots.UniversalRobotsUR10E()
    robot.connect(ip=ip)

    # Read generalized momentum and report the result
    try:
        logger.success(f"Momentum: {robot.get_actual_momentum()}")

    # Ensure we stop the robot and disconnect even if there was an error
    finally:
        robot.disconnect()


def get_example_dict(ip: str):
    return {
        "get_mass_matrix": lambda: get_mass_matrix_example(ip),
        "get_coriolis_and_centrifugal_torques": lambda: get_coriolis_and_centrifugal_torques_example(ip),
        "get_actual_momentum": lambda: get_actual_momentum_example(ip),
    }


def main():
    """
    Run a Dynamics getter Synapse example.
    Usage:
        python get_dynamics_states.py --list
        python get_dynamics_states.py --ip <ROBOT_IP> --example <NAME>
        python get_dynamics_states.py --ip <ROBOT_IP> --all

    Use --list to print the names of available examples without connecting to
    a robot, so you can choose one to pass to --example. --ip is not required
    in this mode because no hardware is contacted.
    """

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Dynamics getter Synapse examples")
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
