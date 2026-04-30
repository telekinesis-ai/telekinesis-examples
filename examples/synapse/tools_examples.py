"""Tools examples for the Synapse SDK on Universal Robots UR10e."""

import argparse
import difflib

from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots
from telekinesis.synapse.tools.parallel_grippers import robotiq


def gripper_connect_disconnect_example(ip: str = "192.168.1.2"):
    """Connect to a Robotiq 2F-85 and cleanly disconnect."""
    logger.info("Running gripper_connect_disconnect example...")
    gripper = robotiq.Robotiq2F85()
    logger.info(f"Connecting Robotiq at {ip}...")
    gripper.connect(ip=ip)
    logger.success("Connected.")
    gripper.disconnect()
    logger.success("Disconnected.")


def gripper_set_unit_example(ip: str = "192.168.1.2"):
    """Switch position unit to normalized (0..1)."""
    logger.info("Running gripper_set_unit example...")
    gripper = robotiq.Robotiq2F85()
    gripper.connect(ip=ip)
    try:
        gripper.set_unit(parameter='position', unit='normalized')
        logger.success("Position unit set to 'normalized'.")

    finally:
        gripper.disconnect()


def gripper_set_position_range_example(ip: str = "192.168.1.2"):
    """Set the gripper stroke range to 85 mm (2F-85 max)."""
    logger.info("Running gripper_set_position_range example...")
    gripper = robotiq.Robotiq2F85()
    gripper.connect(ip=ip)
    try:
        gripper.set_position_range_mm(85.0)
        logger.success("Position range set to 85 mm.")

    finally:
        gripper.disconnect()


def gripper_set_speed_example(ip: str = "192.168.1.2"):
    """Set default speed to 50% of max."""
    logger.info("Running gripper_set_speed example...")
    gripper = robotiq.Robotiq2F85()
    gripper.connect(ip=ip)
    try:
        actual = gripper.set_speed(50.0)
        logger.success(f"Default speed set; effective: {actual}")

    finally:
        gripper.disconnect()


def gripper_set_force_example(ip: str = "192.168.1.2"):
    """Set default grip force to 50%."""
    logger.info("Running gripper_set_force example...")
    gripper = robotiq.Robotiq2F85()
    gripper.connect(ip=ip)
    try:
        actual = gripper.set_force(50.0)
        logger.success(f"Default force set; effective: {actual}")

    finally:
        gripper.disconnect()


def gripper_open_example(ip: str = "192.168.1.2"):
    """Open the gripper fully at 100% speed and 50% force."""
    logger.info("Running gripper_open example...")
    gripper = robotiq.Robotiq2F85()
    gripper.connect(ip=ip)
    try:
        status = gripper.open(speed=100.0, force=50.0, asynchronous=False)
        logger.success(f"open() status: {status}, position: {gripper.get_current_position():.2f}")

    finally:
        gripper.disconnect()


def gripper_close_example(ip: str = "192.168.1.2"):
    """Close the gripper fully at 100% speed and 50% force."""
    logger.info("Running gripper_close example...")
    gripper = robotiq.Robotiq2F85()
    gripper.connect(ip=ip)
    try:
        status = gripper.close(speed=100.0, force=50.0, asynchronous=False)
        logger.success(f"close() status: {status}, position: {gripper.get_current_position():.2f}")

    finally:
        gripper.disconnect()


def gripper_move_example(ip: str = "192.168.1.2"):
    """Move the gripper to 20 mm at 100% speed and 50% force."""
    logger.info("Running gripper_move example...")
    gripper = robotiq.Robotiq2F85()
    gripper.connect(ip=ip)
    try:
        gripper.set_unit(parameter='position', unit='mm')
        gripper.set_position_range_mm(85.0)
        status = gripper.move(position=20.0, speed=100.0, force=50.0, asynchronous=False)
        logger.success(f"move() status: {status}, position: {gripper.get_current_position():.2f}")

    finally:
        gripper.disconnect()


def gripper_get_current_position_example(ip: str = "192.168.1.2"):
    """Read the gripper's current position in the configured unit."""
    logger.info("Running gripper_get_current_position example...")
    gripper = robotiq.Robotiq2F85()
    gripper.connect(ip=ip)
    try:
        logger.success(f"Current position: {gripper.get_current_position()}")

    finally:
        gripper.disconnect()


def get_example_dict(ip: str = "192.168.1.2"):
    return {
        "gripper_connect_disconnect": lambda: gripper_connect_disconnect_example(ip),
        "gripper_set_unit": lambda: gripper_set_unit_example(ip),
        "gripper_set_position_range": lambda: gripper_set_position_range_example(ip),
        "gripper_set_speed": lambda: gripper_set_speed_example(ip),
        "gripper_set_force": lambda: gripper_set_force_example(ip),
        "gripper_open": lambda: gripper_open_example(ip),
        "gripper_close": lambda: gripper_close_example(ip),
        "gripper_move": lambda: gripper_move_example(ip),
        "gripper_get_current_position": lambda: gripper_get_current_position_example(ip),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Tools Synapse examples")
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
