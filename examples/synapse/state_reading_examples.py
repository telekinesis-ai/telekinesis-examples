"""State Reading examples for the Synapse SDK on Universal Robots UR10e."""

import argparse
import difflib

from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots


def get_joint_positions_example(ip: str = "192.168.1.2"):
    """Read current joint positions (degrees)."""
    logger.info("Running get_joint_positions example...")
    robot = universal_robots.UniversalRobotsUR10E()
    robot.connect(ip=ip)
    try:
        q = robot.get_joint_positions()
        logger.success(f"Joint positions: {q}")

    finally:
        robot.disconnect()


def get_cartesian_pose_example(ip: str = "192.168.1.2"):
    """Read current TCP pose [x, y, z, rx, ry, rz]."""
    logger.info("Running get_cartesian_pose example...")
    robot = universal_robots.UniversalRobotsUR10E()
    robot.connect(ip=ip)
    try:
        pose = robot.get_cartesian_pose()
        logger.success(f"TCP pose: {pose}")

    finally:
        robot.disconnect()


def get_pose_transform_example(ip: str = "192.168.1.2"):
    """Apply a 10-cm relative X transform to the current TCP pose."""
    logger.info("Running get_pose_transform example...")
    robot = universal_robots.UniversalRobotsUR10E()
    robot.connect(ip=ip)
    try:
        p_from = robot.get_cartesian_pose()
        p_rel = [0.1, 0.0, 0.0, 0.0, 0.0, 0.0]
        result = robot.get_pose_transform(source_pose=p_from, relative_transform=p_rel)
        logger.success(f"Transformed pose: {result}")

    finally:
        robot.disconnect()


def get_example_dict(ip: str = "192.168.1.2"):
    return {
        "get_joint_positions": lambda: get_joint_positions_example(ip),
        "get_cartesian_pose": lambda: get_cartesian_pose_example(ip),
        "get_pose_transform": lambda: get_pose_transform_example(ip),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="State Reading Synapse examples")
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
