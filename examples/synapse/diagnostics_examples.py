"""Diagnostics examples for the Synapse SDK on Universal Robots UR10e."""

import argparse
import difflib

from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots


def get_actual_execution_time_example(ip: str = "192.168.1.2"):
    """Read controller execution time [s]."""
    logger.info("Running get_actual_execution_time example...")
    robot = universal_robots.UniversalRobotsUR10E()
    robot.connect(ip=ip)
    try:
        logger.success(f"Execution time: {robot.get_actual_execution_time()} s")

    finally:
        robot.disconnect()


def get_actual_tool_accelerometer_example(ip: str = "192.168.1.2"):
    """Read tool-mounted accelerometer reading."""
    logger.info("Running get_actual_tool_accelerometer example...")
    robot = universal_robots.UniversalRobotsUR10E()
    robot.connect(ip=ip)
    try:
        logger.success(f"Tool accel: {robot.get_actual_tool_accelerometer()}")

    finally:
        robot.disconnect()


def get_actual_momentum_example(ip: str = "192.168.1.2"):
    """Read actual momentum estimate."""
    logger.info("Running get_actual_momentum example...")
    robot = universal_robots.UniversalRobotsUR10E()
    robot.connect(ip=ip)
    try:
        logger.success(f"Momentum: {robot.get_actual_momentum()}")

    finally:
        robot.disconnect()


def get_speed_scaling_combined_example(ip: str = "192.168.1.2"):
    """Read combined speed-scaling factor."""
    logger.info("Running get_speed_scaling_combined example...")
    robot = universal_robots.UniversalRobotsUR10E()
    robot.connect(ip=ip)
    try:
        logger.success(f"Speed scaling combined: {robot.get_speed_scaling_combined()}")

    finally:
        robot.disconnect()


def get_example_dict(ip: str = "192.168.1.2"):
    return {
        "get_actual_execution_time": lambda: get_actual_execution_time_example(ip),
        "get_actual_tool_accelerometer": lambda: get_actual_tool_accelerometer_example(ip),
        "get_actual_momentum": lambda: get_actual_momentum_example(ip),
        "get_speed_scaling_combined": lambda: get_speed_scaling_combined_example(ip),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Diagnostics Synapse examples")
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
