"""
Simple Inverse Kinematics examples for the Synapse SDK with default seed.

``inverse_kinematics`` is defined on the abstract manipulator and supports
all robot.

Universal Robots (UR10e) is used here purely for illustration.
The solver runs purely on the kinematic model, so these examples do not
connect to hardware and no ``--ip`` is required.

Usage:
    python inverse_kinematics_example.py
"""

from loguru import logger
from telekinesis.synapse.robots.manipulators import universal_robots

def main():
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


if __name__ == "__main__":
    main()
