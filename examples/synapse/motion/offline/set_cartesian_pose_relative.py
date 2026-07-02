"""
Set Cartesian Pose (relative) example for the Synapse SDK -- offline.

Reads the current TCP pose and moves to a target defined *relative* to it, on the
kinematic model; no hardware connection is made.

Supports all robots.

Usage:
    python set_cartesian_pose_relative.py
"""

from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots


def main():
    """Move the TCP to a target Cartesian pose on the kinematic model."""

    # Create robot instance
    robot = universal_robots.UniversalRobotsUR10E()
    
    # Define target Cartesian pose
    current_cartesian_pose = robot.get_cartesian_pose()
    target_cartesian_pose = current_cartesian_pose.copy()
    target_cartesian_pose[2] += 0.1  # Move 10 cm up in Z

    # Command the move
    robot.set_cartesian_pose(
        cartesian_pose=target_cartesian_pose,
        speed=0.1,
        acceleration=0.1,
    )
    logger.info(f"Moved to target Cartesian pose: {target_cartesian_pose}")


if __name__ == "__main__":
    main()
