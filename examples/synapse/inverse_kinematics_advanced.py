"""
Inverse Kinematics examples for the Synapse SDK.

``inverse_kinematics`` is defined on the abstract manipulator and supports
all robot.

Universal Robots (UR10e) is used here purely for illustration.
The solver runs purely on the kinematic model, so these examples do not
connect to hardware and no ``--ip`` is required.

Usage:
    python inverse_kinematics_example.py --list
    python inverse_kinematics_example.py --example <NAME>
    python inverse_kinematics_example.py --all
"""

import argparse
import difflib

import numpy as np
from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots


def inverse_kinematics_example():
    """Solve IK for a target TCP pose with default solver parameters. Supports all robots."""

    # Create the robot (no connect required — IK runs on the kinematic model)
    robot = universal_robots.UniversalRobotsUR10E()

    # Solve IK for a fixed target pose [x, y, z, rx, ry, rz] (m, deg)
    target_pose = [0.3, 0.3, 0.3, 180, 0, 0]
    try:
        q = robot.inverse_kinematics(target_pose=target_pose)
        logger.success(f"IK solution: {q}")
    except (RuntimeError, TypeError, ValueError) as e:
        logger.error(f"IK failed: {type(e).__name__}: {e}")


def inverse_kinematics_with_seed_example():
    """Solve IK with an explicit initial joint seed. Supports all robots."""

    # Create the robot (no connect required — IK runs on the kinematic model)
    robot = universal_robots.UniversalRobotsUR10E()

    # Pick a known-reachable configuration and derive its forward-kinematic pose
    q_deg = np.asarray([0, -90 - 5.7, -90, 0, 90, 0], dtype=float)
    target_pose = robot.forward_kinematics(q_deg)

    # Perturb the configuration to use as an IK seed
    q_init = q_deg + np.random.normal(loc=0.0, scale=30.0, size=q_deg.shape)

    # Solve IK with the explicit seed and report the result
    try:
        q = robot.inverse_kinematics(
            target_pose=target_pose,
            q_init=q_init,
            solver="multi_start_clik",
        )
        logger.success(f"IK solution: {q}")
    except (RuntimeError, TypeError, ValueError) as e:
        logger.error(f"IK failed: {type(e).__name__}: {e}")


def inverse_kinematics_with_profile_example():
    """Solve IK with ``profile=True`` and log the timing diagnostics. Supports all robots."""

    # Create the robot (no connect required — IK runs on the kinematic model)
    robot = universal_robots.UniversalRobotsUR10E()

    # Solve IK with profiling enabled to return (q, timing) and log both
    target_pose = [0.5, 0.2, 0.3, 180.0, 0.0, 0.0]
    try:
        q, timing = robot.inverse_kinematics(
            target_pose=target_pose,
            profile=True,
        )
        logger.success(f"IK solution: {q}")
        logger.info(f"Total time: {timing['total_s']:.4f} s")
        logger.info(f"Seeds tried: {timing['num_seeds_tried']}")
        logger.info(f"Winning seed: {timing['winning_seed_index']}")
        logger.info(f"Linear error (m):  {timing['linear_error_norm_meters']:.6f}")
        logger.info(f"Angular error (rad): {timing['angular_error_norm_rad']:.6f}")
    except (RuntimeError, TypeError, ValueError) as e:
        logger.error(f"IK failed: {type(e).__name__}: {e}")


def get_example_dict():
    return {
        "inverse_kinematics": inverse_kinematics_example,
        "inverse_kinematics_with_seed": inverse_kinematics_with_seed_example,
        "inverse_kinematics_with_profile": inverse_kinematics_with_profile_example,
    }


def main():
    """
    Run an Inverse Kinematics Synapse example.
    Usage:
        python inverse_kinematics_example.py --list
        python inverse_kinematics_example.py --example <NAME>
        python inverse_kinematics_example.py --all
    """

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Inverse Kinematics Synapse examples")
    parser.add_argument("--example", type=str)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()
    examples = get_example_dict()

    # Handle example selection
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
        return

    # Handle single example execution
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
