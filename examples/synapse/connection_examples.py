"""Connection examples for the Synapse SDK on Universal Robots UR10e."""

import argparse
import difflib

from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots


def connect_and_disconnect_example(ip: str = "192.168.1.2"):
    """Connect to the robot, verify the link, and cleanly disconnect."""
    logger.info("Running connect_and_disconnect example...")
    robot = universal_robots.UniversalRobotsUR10E()
    logger.info(f"Connecting to UR10e at {ip}...")
    robot.connect(ip=ip)
    logger.success("Connected.")
    robot.disconnect()
    logger.success("Disconnected.")


def get_example_dict(ip: str = "192.168.1.2"):
    return {
        "connect_and_disconnect": lambda: connect_and_disconnect_example(ip),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Connection Synapse examples")
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
