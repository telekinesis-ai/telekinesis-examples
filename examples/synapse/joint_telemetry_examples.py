"""Joint Telemetry examples for the Synapse SDK on Universal Robots UR10e."""

import argparse
import difflib

from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots


def get_actual_joint_velocities_example(ip: str = "192.168.1.2"):
    """Read actual joint velocities [deg/s]."""
    logger.info("Running get_actual_joint_velocities example...")
    robot = universal_robots.UniversalRobotsUR10E()
    robot.connect(ip=ip)
    try:
        logger.success(f"Joint velocities: {robot.get_actual_joint_velocities()}")

    finally:
        robot.disconnect()


def get_actual_joint_currents_example(ip: str = "192.168.1.2"):
    """Read actual joint currents [A]."""
    logger.info("Running get_actual_joint_currents example...")
    robot = universal_robots.UniversalRobotsUR10E()
    robot.connect(ip=ip)
    try:
        logger.success(f"Joint currents: {robot.get_actual_joint_currents()}")

    finally:
        robot.disconnect()


def get_joint_temperatures_example(ip: str = "192.168.1.2"):
    """Read joint temperatures [°C]."""
    logger.info("Running get_joint_temperatures example...")
    robot = universal_robots.UniversalRobotsUR10E()
    robot.connect(ip=ip)
    try:
        logger.success(f"Joint temperatures: {robot.get_joint_temperatures()}")

    finally:
        robot.disconnect()


def get_joint_torques_example(ip: str = "192.168.1.2"):
    """Read joint torques [N·m]."""
    logger.info("Running get_joint_torques example...")
    robot = universal_robots.UniversalRobotsUR10E()
    robot.connect(ip=ip)
    try:
        logger.success(f"Joint torques: {robot.get_joint_torques()}")

    finally:
        robot.disconnect()


def get_actual_joint_positions_history_example(ip: str = "192.168.1.2"):
    """Read historical joint positions (steps=0 = most recent)."""
    logger.info("Running get_actual_joint_positions_history example...")
    robot = universal_robots.UniversalRobotsUR10E()
    robot.connect(ip=ip)
    try:
        logger.success(f"Joint positions history: {robot.get_actual_joint_positions_history(steps=0)}")

    finally:
        robot.disconnect()


def get_example_dict(ip: str = "192.168.1.2"):
    return {
        "get_actual_joint_velocities": lambda: get_actual_joint_velocities_example(ip),
        "get_actual_joint_currents": lambda: get_actual_joint_currents_example(ip),
        "get_joint_temperatures": lambda: get_joint_temperatures_example(ip),
        "get_joint_torques": lambda: get_joint_torques_example(ip),
        "get_actual_joint_positions_history": lambda: get_actual_joint_positions_history_example(ip),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Joint Telemetry Synapse examples")
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
