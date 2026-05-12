"""
Read the Pinocchio kinematic model for the Synapse SDK.

``get_model`` returns the standard ``pinocchio.Model`` object built from
the robot's URDF. This is the Python Pinocchio model — distinct from the
internal C++ Synapse wrapper used by IK/FK methods — and is useful for
custom kinematics, dynamics, or downstream Pinocchio-based pipelines.

Universal Robots (UR10e) is used here purely for illustration.
This example runs purely on the kinematic model and does not connect to
hardware — no ``--ip`` is required.

Usage:
    python get_model.py
"""

from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots


def main():
    """Read the Pinocchio kinematic model and log a few summary fields."""

    # Create the robot (no connect required — runs on the kinematic model)
    robot = universal_robots.UniversalRobotsUR10E()

    # Read the Pinocchio kinematic model
    model = robot.get_model()
    logger.success(f"Model: {model}")
    logger.info(f"Model name: {model.name}")
    logger.info(f"nq (configuration dim): {model.nq}")
    logger.info(f"nv (velocity dim): {model.nv}")
    logger.info(f"Number of joints: {model.njoints}")
    logger.info(f"Number of frames: {model.nframes}")


if __name__ == "__main__":
    main()
