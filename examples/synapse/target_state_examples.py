"""Target State examples for the Synapse SDK on Universal Robots UR10e."""

import argparse
import difflib

from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots


def get_target_joint_positions_example(ip: str = "192.168.1.2"):
    """Read target joint positions [deg]."""
    logger.info("Running get_target_joint_positions example...")
    robot = universal_robots.UniversalRobotsUR10E()
    robot.connect(ip=ip)
    try:
        logger.success(f"Target q: {robot.get_target_joint_positions()}")

    finally:
        robot.disconnect()


def get_target_joint_velocities_example(ip: str = "192.168.1.2"):
    """Read target joint velocities [deg/s]."""
    logger.info("Running get_target_joint_velocities example...")
    robot = universal_robots.UniversalRobotsUR10E()
    robot.connect(ip=ip)
    try:
        logger.success(f"Target qd: {robot.get_target_joint_velocities()}")

    finally:
        robot.disconnect()


def get_target_joint_currents_example(ip: str = "192.168.1.2"):
    """Read target joint currents [A]."""
    logger.info("Running get_target_joint_currents example...")
    robot = universal_robots.UniversalRobotsUR10E()
    robot.connect(ip=ip)
    try:
        logger.success(f"Target currents: {robot.get_target_joint_currents()}")

    finally:
        robot.disconnect()


def get_target_joint_moments_example(ip: str = "192.168.1.2"):
    """Read target joint moments [N·m]."""
    logger.info("Running get_target_joint_moments example...")
    robot = universal_robots.UniversalRobotsUR10E()
    robot.connect(ip=ip)
    try:
        logger.success(f"Target moments: {robot.get_target_joint_moments()}")

    finally:
        robot.disconnect()


def get_target_joint_accelerations_example(ip: str = "192.168.1.2"):
    """Read target joint accelerations [deg/s²]."""
    logger.info("Running get_target_joint_accelerations example...")
    robot = universal_robots.UniversalRobotsUR10E()
    robot.connect(ip=ip)
    try:
        logger.success(f"Target qdd: {robot.get_target_joint_accelerations()}")

    finally:
        robot.disconnect()


def get_target_tcp_pose_example(ip: str = "192.168.1.2"):
    """Read target TCP pose."""
    logger.info("Running get_target_tcp_pose example...")
    robot = universal_robots.UniversalRobotsUR10E()
    robot.connect(ip=ip)
    try:
        logger.success(f"Target TCP pose: {robot.get_target_tcp_pose()}")

    finally:
        robot.disconnect()


def get_target_tcp_speed_example(ip: str = "192.168.1.2"):
    """Read target TCP speed."""
    logger.info("Running get_target_tcp_speed example...")
    robot = universal_robots.UniversalRobotsUR10E()
    robot.connect(ip=ip)
    try:
        logger.success(f"Target TCP speed: {robot.get_target_tcp_speed()}")

    finally:
        robot.disconnect()


def get_target_speed_fraction_example(ip: str = "192.168.1.2"):
    """Read target speed fraction (0..1)."""
    logger.info("Running get_target_speed_fraction example...")
    robot = universal_robots.UniversalRobotsUR10E()
    robot.connect(ip=ip)
    try:
        logger.success(f"Target speed fraction: {robot.get_target_speed_fraction()}")

    finally:
        robot.disconnect()


def get_example_dict(ip: str = "192.168.1.2"):
    return {
        "get_target_joint_positions": lambda: get_target_joint_positions_example(ip),
        "get_target_joint_velocities": lambda: get_target_joint_velocities_example(ip),
        "get_target_joint_currents": lambda: get_target_joint_currents_example(ip),
        "get_target_joint_moments": lambda: get_target_joint_moments_example(ip),
        "get_target_joint_accelerations": lambda: get_target_joint_accelerations_example(ip),
        "get_target_tcp_pose": lambda: get_target_tcp_pose_example(ip),
        "get_target_tcp_speed": lambda: get_target_tcp_speed_example(ip),
        "get_target_speed_fraction": lambda: get_target_speed_fraction_example(ip),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Target State Synapse examples")
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
