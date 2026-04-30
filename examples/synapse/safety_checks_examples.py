"""Safety Checks examples for the Synapse SDK on Universal Robots UR10e."""

import argparse
import difflib

from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots


def is_protective_stopped_example(ip: str = "192.168.1.2"):
    """Check whether the robot is in a protective stop."""
    logger.info("Running is_protective_stopped example...")
    robot = universal_robots.UniversalRobotsUR10E()
    robot.connect(ip=ip)
    try:
        logger.success(f"Is protective stopped: {robot.is_protective_stopped()}")

    finally:
        robot.disconnect()


def is_emergency_stopped_example(ip: str = "192.168.1.2"):
    """Check whether the robot is in an emergency stop."""
    logger.info("Running is_emergency_stopped example...")
    robot = universal_robots.UniversalRobotsUR10E()
    robot.connect(ip=ip)
    try:
        logger.success(f"Is emergency stopped: {robot.is_emergency_stopped()}")

    finally:
        robot.disconnect()


def is_program_running_example(ip: str = "192.168.1.2"):
    """Check whether a program is currently running on the controller."""
    logger.info("Running is_program_running example...")
    robot = universal_robots.UniversalRobotsUR10E()
    robot.connect(ip=ip)
    try:
        logger.success(f"Program running: {robot.is_program_running_on_controller()}")

    finally:
        robot.disconnect()


def is_steady_example(ip: str = "192.168.1.2"):
    """Check whether the robot is currently motionless."""
    logger.info("Running is_steady example...")
    robot = universal_robots.UniversalRobotsUR10E()
    robot.connect(ip=ip)
    try:
        logger.success(f"Is steady: {robot.is_steady()}")

    finally:
        robot.disconnect()


def is_pose_within_safety_limits_example(ip: str = "192.168.1.2"):
    """Check whether a candidate pose is within the configured safety limits."""
    logger.info("Running is_pose_within_safety_limits example...")
    robot = universal_robots.UniversalRobotsUR10E()
    robot.connect(ip=ip)
    try:
        pose = [0.5, 0.0, 0.5, 0.0, 0.0, 0.0]
        logger.success(f"Pose within limits: {robot.is_pose_within_safety_limits(pose)}")

    finally:
        robot.disconnect()


def is_joints_within_safety_limits_example(ip: str = "192.168.1.2"):
    """Check whether a candidate joint configuration is within the configured safety limits."""
    logger.info("Running is_joints_within_safety_limits example...")
    robot = universal_robots.UniversalRobotsUR10E()
    robot.connect(ip=ip)
    try:
        q = [0.0, -90.0, -90.0, 0.0, 90.0, 0.0]
        logger.success(f"Joints within limits: {robot.is_joints_within_safety_limits(q)}")

    finally:
        robot.disconnect()


def unlock_protective_stop_example(ip: str = "192.168.1.2"):
    """Unlock a protective stop (only valid after the cause has cleared)."""
    logger.info("Running unlock_protective_stop example...")
    robot = universal_robots.UniversalRobotsUR10E()
    robot.connect(ip=ip)
    try:
        robot.unlock_protective_stop()
        logger.success("Protective stop unlocked.")

    finally:
        robot.disconnect()


def get_safety_status_bits_example(ip: str = "192.168.1.2"):
    """Read the safety-status bitmask."""
    logger.info("Running get_safety_status_bits example...")
    robot = universal_robots.UniversalRobotsUR10E()
    robot.connect(ip=ip)
    try:
        logger.success(f"Safety bits: {robot.get_safety_status_bits():#013b}")

    finally:
        robot.disconnect()


def get_example_dict(ip: str = "192.168.1.2"):
    return {
        "is_protective_stopped": lambda: is_protective_stopped_example(ip),
        "is_emergency_stopped": lambda: is_emergency_stopped_example(ip),
        "is_program_running": lambda: is_program_running_example(ip),
        "is_steady": lambda: is_steady_example(ip),
        "is_pose_within_safety_limits": lambda: is_pose_within_safety_limits_example(ip),
        "is_joints_within_safety_limits": lambda: is_joints_within_safety_limits_example(ip),
        "unlock_protective_stop": lambda: unlock_protective_stop_example(ip),
        "get_safety_status_bits": lambda: get_safety_status_bits_example(ip),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Safety Checks Synapse examples")
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
