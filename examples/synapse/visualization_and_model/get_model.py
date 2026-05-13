"""
Read the Pinocchio kinematic model for the Synapse SDK.

``get_model`` returns the standard ``pinocchio.Model`` object built from
the robot's URDF. 

Universal Robots (UR10e) is used here purely for illustration. It supports all robots.

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
