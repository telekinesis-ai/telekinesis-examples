"""TCP Telemetry examples for the Synapse SDK on Universal Robots UR10e."""

import argparse
import difflib

from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots


def get_actual_tcp_speed_example(ip: str = "192.168.1.2"):
    """Read TCP speed [vx, vy, vz, vrx, vry, vrz]."""
    logger.info("Running get_actual_tcp_speed example...")
    robot = universal_robots.UniversalRobotsUR10E()
    robot.connect(ip=ip)
    try:
        logger.success(f"TCP speed: {robot.get_actual_tcp_speed()}")

    finally:
        robot.disconnect()


def get_actual_tcp_force_example(ip: str = "192.168.1.2"):
    """Read TCP force/torque [Fx, Fy, Fz, Mx, My, Mz]."""
    logger.info("Running get_actual_tcp_force example...")
    robot = universal_robots.UniversalRobotsUR10E()
    robot.connect(ip=ip)
    try:
        logger.success(f"TCP force: {robot.get_actual_tcp_force()}")

    finally:
        robot.disconnect()


def get_target_waypoint_example(ip: str = "192.168.1.2"):
    """Read the current target waypoint."""
    logger.info("Running get_target_waypoint example...")
    robot = universal_robots.UniversalRobotsUR10E()
    robot.connect(ip=ip)
    try:
        logger.success(f"Target waypoint: {robot.get_target_waypoint()}")

    finally:
        robot.disconnect()


def get_tcp_offset_example(ip: str = "192.168.1.2"):
    """Read the active TCP offset."""
    logger.info("Running get_tcp_offset example...")
    robot = universal_robots.UniversalRobotsUR10E()
    robot.connect(ip=ip)
    try:
        logger.success(f"TCP offset: {robot.get_tcp_offset()}")

    finally:
        robot.disconnect()


def get_ft_raw_wrench_example(ip: str = "192.168.1.2"):
    """Read raw F/T sensor wrench."""
    logger.info("Running get_ft_raw_wrench example...")
    robot = universal_robots.UniversalRobotsUR10E()
    robot.connect(ip=ip)
    try:
        logger.success(f"FT raw wrench: {robot.get_ft_raw_wrench()}")

    finally:
        robot.disconnect()


def get_example_dict(ip: str = "192.168.1.2"):
    return {
        "get_actual_tcp_speed": lambda: get_actual_tcp_speed_example(ip),
        "get_actual_tcp_force": lambda: get_actual_tcp_force_example(ip),
        "get_target_waypoint": lambda: get_target_waypoint_example(ip),
        "get_tcp_offset": lambda: get_tcp_offset_example(ip),
        "get_ft_raw_wrench": lambda: get_ft_raw_wrench_example(ip),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="TCP Telemetry Synapse examples")
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
