"""Joint Detail examples for the Synapse SDK on Universal Robots UR10e."""

import argparse
import difflib

from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots


def get_joint_control_output_example(ip: str = "192.168.1.2"):
    """Read joint control-output currents [A]."""
    logger.info("Running get_joint_control_output example...")
    robot = universal_robots.UniversalRobotsUR10E()
    robot.connect(ip=ip)
    try:
        logger.success(f"Joint control output: {robot.get_joint_control_output()}")

    finally:
        robot.disconnect()


def get_joint_mode_example(ip: str = "192.168.1.2"):
    """Read per-joint control modes."""
    logger.info("Running get_joint_mode example...")
    robot = universal_robots.UniversalRobotsUR10E()
    robot.connect(ip=ip)
    try:
        logger.success(f"Joint modes: {robot.get_joint_mode()}")

    finally:
        robot.disconnect()


def get_payload_inertia_example(ip: str = "192.168.1.2"):
    """Read payload inertia tensor [Ixx, Iyy, Izz, Ixy, Ixz, Iyz]."""
    logger.info("Running get_payload_inertia example...")
    robot = universal_robots.UniversalRobotsUR10E()
    robot.connect(ip=ip)
    try:
        logger.success(f"Payload inertia: {robot.get_payload_inertia()}")

    finally:
        robot.disconnect()


def get_example_dict(ip: str = "192.168.1.2"):
    return {
        "get_joint_control_output": lambda: get_joint_control_output_example(ip),
        "get_joint_mode": lambda: get_joint_mode_example(ip),
        "get_payload_inertia": lambda: get_payload_inertia_example(ip),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Joint Detail Synapse examples")
    parser.add_argument("--ip", type=str, default="192.168.1.2")
    parser.add_argument("--example", type=str)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--pause", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    examples = get_example_dict(args.ip)
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
            if args.pause:
                input("Press Enter for next example...")
        return
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
