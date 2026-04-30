"""Servo Control examples for the Synapse SDK on Universal Robots UR10e."""

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

def servo_joint_example(ip: str = "192.168.1.2"):
    """Hold the current joint configuration via servo_joint for 1 second."""
    logger.info("Running servo_joint example...")
    robot = universal_robots.UniversalRobotsUR10E()
    robot.connect(ip=ip)
    try:
        q = robot.get_joint_positions()
        _confirm_real_motion(f"Will stream servo_joint hold at {q} for 1 s")
        deadline = time.monotonic() + 1.0
        dt = 0.002
        while time.monotonic() < deadline:
            robot.servo_joint(q=q, speed=0.1, acceleration=0.1, time=dt, lookahead_time=0.1, gain=300)
            time.sleep(dt)
        robot.servo_stop()
        logger.success("servo_joint loop complete.")

    finally:
        robot.disconnect()


def servo_cartesian_example(ip: str = "192.168.1.2"):
    """Hold the current Cartesian pose via servo_cartesian for 1 second."""
    logger.info("Running servo_cartesian example...")
    robot = universal_robots.UniversalRobotsUR10E()
    robot.connect(ip=ip)
    try:
        pose = robot.get_cartesian_pose()
        _confirm_real_motion(f"Will stream servo_cartesian hold at {pose} for 1 s")
        deadline = time.monotonic() + 1.0
        dt = 0.002
        while time.monotonic() < deadline:
            robot.servo_cartesian(pose=pose, speed=0.1, acceleration=0.1, time=dt, lookahead_time=0.1, gain=300)
            time.sleep(dt)
        robot.servo_stop()
        logger.success("servo_cartesian loop complete.")

    finally:
        robot.disconnect()


def servo_circular_example(ip: str = "192.168.1.2"):
    """Send a circular servo command via a 2-cm via-point above the current pose."""
    logger.info("Running servo_circular example...")
    robot = universal_robots.UniversalRobotsUR10E()
    robot.connect(ip=ip)
    try:
        current = robot.get_cartesian_pose()
        via = list(current); via[2] += 0.02
        _confirm_real_motion(f"Will execute servo_circular through via-point {via}")
        robot.servo_circular(pose=via, speed=0.05, acceleration=0.24, blend=0.0)
        robot.servo_stop()
        logger.success("servo_circular done.")

    finally:
        robot.disconnect()


def servo_stop_example(ip: str = "192.168.1.2"):
    """Stop any active servo command (safe when idle)."""
    logger.info("Running servo_stop example...")
    robot = universal_robots.UniversalRobotsUR10E()
    robot.connect(ip=ip)
    try:
        robot.servo_stop(deceleration=10.0)
        logger.success("Servo stopped.")

    finally:
        robot.disconnect()


def get_example_dict(ip: str = "192.168.1.2"):
    return {
        "servo_joint": lambda: servo_joint_example(ip),
        "servo_cartesian": lambda: servo_cartesian_example(ip),
        "servo_circular": lambda: servo_circular_example(ip),
        "servo_stop": lambda: servo_stop_example(ip),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Servo Control Synapse examples")
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
