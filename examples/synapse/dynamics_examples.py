"""Dynamics examples for the Synapse SDK on Universal Robots UR10e."""

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

def set_gravity_example(ip: str = "192.168.1.2"):
    """Set gravity to standard upright mounting (Z=9.82)."""
    logger.info("Running set_gravity example...")
    robot = universal_robots.UniversalRobotsUR10E()
    robot.connect(ip=ip)
    try:
        robot.set_gravity([0.0, 0.0, 9.82])
        logger.success("Gravity set.")

    finally:
        robot.disconnect()


def set_target_payload_example(ip: str = "192.168.1.2"):
    """Set target payload with mass, CoG, and full inertia tensor."""
    logger.info("Running set_target_payload example...")
    robot = universal_robots.UniversalRobotsUR10E()
    robot.connect(ip=ip)
    try:
        robot.set_target_payload(mass=1.5, cog=[0.0, 0.0, 0.05], inertia=[0.001, 0.001, 0.0005, 0.0, 0.0, 0.0])
        logger.success("Target payload set.")

    finally:
        robot.disconnect()


def direct_torque_example(ip: str = "192.168.1.2"):
    """Send 10 zero-torque commands at the controller step rate."""
    logger.info("Running direct_torque example...")
    robot = universal_robots.UniversalRobotsUR10E()
    robot.connect(ip=ip)
    try:
        _confirm_real_motion("Will send 10 direct-torque commands (zero torques)")
        dt = robot.get_step_time()
        for _ in range(10):
            robot.direct_torque([0.0]*6)
            time.sleep(dt)
        logger.success("Direct-torque commands sent.")

    finally:
        robot.disconnect()


def get_mass_matrix_example(ip: str = "192.168.1.2"):
    """Read the joint-space mass matrix (6x6 flat)."""
    logger.info("Running get_mass_matrix example...")
    robot = universal_robots.UniversalRobotsUR10E()
    robot.connect(ip=ip)
    try:
        logger.success(f"Mass matrix: {robot.get_mass_matrix()}")

    finally:
        robot.disconnect()


def get_coriolis_and_centrifugal_torques_example(ip: str = "192.168.1.2"):
    """Read Coriolis and centrifugal torques [N·m]."""
    logger.info("Running get_coriolis_and_centrifugal_torques example...")
    robot = universal_robots.UniversalRobotsUR10E()
    robot.connect(ip=ip)
    try:
        logger.success(f"Coriolis/centrifugal torques: {robot.get_coriolis_and_centrifugal_torques()}")

    finally:
        robot.disconnect()


def get_example_dict(ip: str = "192.168.1.2"):
    return {
        "set_gravity": lambda: set_gravity_example(ip),
        "set_target_payload": lambda: set_target_payload_example(ip),
        "direct_torque": lambda: direct_torque_example(ip),
        "get_mass_matrix": lambda: get_mass_matrix_example(ip),
        "get_coriolis_and_centrifugal_torques": lambda: get_coriolis_and_centrifugal_torques_example(ip),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Dynamics Synapse examples")
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
