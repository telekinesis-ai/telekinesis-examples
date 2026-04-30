"""Robot Status examples for the Synapse SDK on Universal Robots UR10e."""

import argparse
import difflib

from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots


def get_timestamp_example(ip: str = "192.168.1.2"):
    """Read current controller timestamp [s]."""
    logger.info("Running get_timestamp example...")
    robot = universal_robots.UniversalRobotsUR10E()
    robot.connect(ip=ip)
    try:
        logger.success(f"Timestamp: {robot.get_timestamp()}")

    finally:
        robot.disconnect()


def get_robot_mode_example(ip: str = "192.168.1.2"):
    """Read robot mode string."""
    logger.info("Running get_robot_mode example...")
    robot = universal_robots.UniversalRobotsUR10E()
    robot.connect(ip=ip)
    try:
        logger.success(f"Robot mode: {robot.get_robot_mode()}")

    finally:
        robot.disconnect()


def get_robot_status_example(ip: str = "192.168.1.2"):
    """Read robot status string."""
    logger.info("Running get_robot_status example...")
    robot = universal_robots.UniversalRobotsUR10E()
    robot.connect(ip=ip)
    try:
        logger.success(f"Robot status: {robot.get_robot_status()}")

    finally:
        robot.disconnect()


def get_safety_mode_example(ip: str = "192.168.1.2"):
    """Read safety mode string."""
    logger.info("Running get_safety_mode example...")
    robot = universal_robots.UniversalRobotsUR10E()
    robot.connect(ip=ip)
    try:
        logger.success(f"Safety mode: {robot.get_safety_mode()}")

    finally:
        robot.disconnect()


def get_runtime_state_example(ip: str = "192.168.1.2"):
    """Read runtime state string."""
    logger.info("Running get_runtime_state example...")
    robot = universal_robots.UniversalRobotsUR10E()
    robot.connect(ip=ip)
    try:
        logger.success(f"Runtime state: {robot.get_runtime_state()}")

    finally:
        robot.disconnect()


def get_controller_frequency_example(ip: str = "192.168.1.2"):
    """Estimate controller frequency over a 0.2-s window."""
    logger.info("Running get_controller_frequency example...")
    robot = universal_robots.UniversalRobotsUR10E()
    robot.connect(ip=ip)
    try:
        logger.success(f"Controller frequency: {robot.get_controller_frequency(window_s=0.2)} Hz")

    finally:
        robot.disconnect()


def get_example_dict(ip: str = "192.168.1.2"):
    return {
        "get_timestamp": lambda: get_timestamp_example(ip),
        "get_robot_mode": lambda: get_robot_mode_example(ip),
        "get_robot_status": lambda: get_robot_status_example(ip),
        "get_safety_mode": lambda: get_safety_mode_example(ip),
        "get_runtime_state": lambda: get_runtime_state_example(ip),
        "get_controller_frequency": lambda: get_controller_frequency_example(ip),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Robot Status Synapse examples")
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
