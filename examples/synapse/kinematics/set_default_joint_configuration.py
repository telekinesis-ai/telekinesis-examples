"""
Set the default joint configuration for the Synapse SDK.

``set_default_joint_configuration`` overrides the brand-default joint configuration
that are used as the offline commanded state. Values are in degrees.

Universal Robots (UR10e) is used here purely for illustration. It supports all robots. This example runs purely on the kinematic model and does

Usage:
    python set_default_joint_configuration.py
"""

from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots


def main():
    """Override the default joint configuration and verify the readback."""

    # Create the robot (no connect required — runs on the kinematic model)
    robot = universal_robots.UniversalRobotsUR10E()

    # Log the brand-default joint configuration [deg]
    logger.info(f"Default joint configuration [deg]: {robot.default_joint_configuration}")

    # Set a new default joint configuration [deg]
    new_default = [0.0, -90.0, -90.0, 0.0, 90.0, 0.0]
    robot.set_default_joint_configuration(new_default)
    logger.success(f"Updated default joint configuration [deg]: {robot.default_joint_configuration}")


if __name__ == "__main__":
    main()
