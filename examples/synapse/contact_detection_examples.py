"""Contact Detection examples for the Synapse SDK on Universal Robots UR10e."""

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

def start_contact_detection_example(ip: str = "192.168.1.2"):
    """Start an async +Z move and arm contact detection; poll for contact for ~1 s."""
    logger.info("Running start_contact_detection example...")
    robot = universal_robots.UniversalRobotsUR10E()
    robot.connect(ip=ip)
    try:
        current = robot.get_cartesian_pose()
        target = list(current); target[2] += 0.02
        _confirm_real_motion(f"Will move async toward {target} and watch for contact")
        robot.set_cartesian_pose(target, speed=0.05, acceleration=0.1, asynchronous=True)
        robot.start_contact_detection()
        for _ in range(20):
            if robot.read_contact_detection():
                logger.info("Contact detected!")
                break
            time.sleep(0.05)
        contact = robot.stop_contact_detection()
        logger.success(f"Final contact: {contact}")

    finally:
        robot.disconnect()


def read_contact_detection_example(ip: str = "192.168.1.2"):
    """Read whether contact has been detected since the last start_contact_detection() call."""
    logger.info("Running read_contact_detection example...")
    robot = universal_robots.UniversalRobotsUR10E()
    robot.connect(ip=ip)
    try:
        robot.start_contact_detection()
        contact = robot.read_contact_detection()
        logger.success(f"Contact: {contact}")
        robot.stop_contact_detection()

    finally:
        robot.disconnect()


def stop_contact_detection_example(ip: str = "192.168.1.2"):
    """Stop contact detection and return the final contact result."""
    logger.info("Running stop_contact_detection example...")
    robot = universal_robots.UniversalRobotsUR10E()
    robot.connect(ip=ip)
    try:
        robot.start_contact_detection()
        result = robot.stop_contact_detection()
        logger.success(f"Stop result: {result}")

    finally:
        robot.disconnect()


def get_example_dict(ip: str = "192.168.1.2"):
    return {
        "start_contact_detection": lambda: start_contact_detection_example(ip),
        "read_contact_detection": lambda: read_contact_detection_example(ip),
        "stop_contact_detection": lambda: stop_contact_detection_example(ip),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Contact Detection Synapse examples")
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
