"""Speed Control examples for the Synapse SDK on Universal Robots UR10e."""

import argparse
import difflib
import os
import time

from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots


def _confirm_real_motion(description: str) -> None:
    """Print a loud warning and require explicit 'yes' to proceed (bypassed by SYNAPSE_EXAMPLES_NONINTERACTIVE=1)."""
    logger.warning("=" * 70)
    logger.warning("REAL ROBOT MOTION ABOUT TO EXECUTE")
    logger.warning(description)
    logger.warning(
        "This command will move physical hardware. You are responsible for ensuring "
        "the workspace is clear, the e-stop is reachable, and the trajectory is safe."
    )
    logger.warning("=" * 70)
    if input("Type 'yes' to proceed: ").strip().lower() != "yes":
        raise SystemExit("Aborted by user.")

def speed_joint_example(ip: str = "192.168.1.2"):
    """Move joint 1 at 1 deg/s for 1 second using joint-velocity control."""
    logger.info("Running speed_joint example...")
    robot = universal_robots.UniversalRobotsUR10E()
    robot.connect(ip=ip)
    try:
        qd = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        _confirm_real_motion(f"Will command joint speed {qd} for 1 s")
        robot.move_with_joint_speed(qd=qd, acceleration=2.0)
        time.sleep(1.0)
        robot.stop_speed_motion()
        logger.success("speed_joint done.")

    finally:
        robot.disconnect()


def speed_cartesian_example(ip: str = "192.168.1.2"):
    """Move TCP at 1 cm/s along base Z for 1 second using Cartesian-velocity control."""
    logger.info("Running speed_cartesian example...")
    robot = universal_robots.UniversalRobotsUR10E()
    robot.connect(ip=ip)
    try:
        xd = [0.0, 0.0, 0.01, 0.0, 0.0, 0.0]
        _confirm_real_motion(f"Will command Cartesian velocity {xd} for 1 s")
        robot.move_with_cartesian_velocity(xd=xd, acceleration=0.05)
        time.sleep(1.0)
        robot.stop_speed_motion()
        logger.success("speed_cartesian done.")

    finally:
        robot.disconnect()


def speed_stop_example(ip: str = "192.168.1.2"):
    """Stop any active speed (velocity) controller."""
    logger.info("Running speed_stop example...")
    robot = universal_robots.UniversalRobotsUR10E()
    robot.connect(ip=ip)
    try:
        robot.stop_speed_motion(deceleration=10.0)
        logger.success("Speed motion stopped.")

    finally:
        robot.disconnect()


def get_example_dict(ip: str = "192.168.1.2"):
    return {
        "speed_joint": lambda: speed_joint_example(ip),
        "speed_cartesian": lambda: speed_cartesian_example(ip),
        "speed_stop": lambda: speed_stop_example(ip),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Speed Control Synapse examples")
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
