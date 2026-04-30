"""Electrical examples for the Synapse SDK on Universal Robots UR10e."""

import argparse
import difflib

from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots


def get_actual_main_voltage_example(ip: str = "192.168.1.2"):
    """Read main voltage [V]."""
    logger.info("Running get_actual_main_voltage example...")
    robot = universal_robots.UniversalRobotsUR10E()
    robot.connect(ip=ip)
    try:
        logger.success(f"Main voltage: {robot.get_actual_main_voltage()} V")

    finally:
        robot.disconnect()


def get_actual_robot_voltage_example(ip: str = "192.168.1.2"):
    """Read robot voltage [V]."""
    logger.info("Running get_actual_robot_voltage example...")
    robot = universal_robots.UniversalRobotsUR10E()
    robot.connect(ip=ip)
    try:
        logger.success(f"Robot voltage: {robot.get_actual_robot_voltage()} V")

    finally:
        robot.disconnect()


def get_actual_robot_current_example(ip: str = "192.168.1.2"):
    """Read robot current [A]."""
    logger.info("Running get_actual_robot_current example...")
    robot = universal_robots.UniversalRobotsUR10E()
    robot.connect(ip=ip)
    try:
        logger.success(f"Robot current: {robot.get_actual_robot_current()} A")

    finally:
        robot.disconnect()


def get_actual_joint_voltage_example(ip: str = "192.168.1.2"):
    """Read joint voltages [V]."""
    logger.info("Running get_actual_joint_voltage example...")
    robot = universal_robots.UniversalRobotsUR10E()
    robot.connect(ip=ip)
    try:
        logger.success(f"Joint voltages: {robot.get_actual_joint_voltage()}")

    finally:
        robot.disconnect()


def get_actual_current_as_torque_example(ip: str = "192.168.1.2"):
    """Read joint currents converted to torques [N·m]."""
    logger.info("Running get_actual_current_as_torque example...")
    robot = universal_robots.UniversalRobotsUR10E()
    robot.connect(ip=ip)
    try:
        logger.success(f"Current-as-torque: {robot.get_actual_current_as_torque()}")

    finally:
        robot.disconnect()


def get_example_dict(ip: str = "192.168.1.2"):
    return {
        "get_actual_main_voltage": lambda: get_actual_main_voltage_example(ip),
        "get_actual_robot_voltage": lambda: get_actual_robot_voltage_example(ip),
        "get_actual_robot_current": lambda: get_actual_robot_current_example(ip),
        "get_actual_joint_voltage": lambda: get_actual_joint_voltage_example(ip),
        "get_actual_current_as_torque": lambda: get_actual_current_as_torque_example(ip),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Electrical Synapse examples")
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
