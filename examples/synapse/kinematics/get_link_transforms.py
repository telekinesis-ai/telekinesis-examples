"""
Per-link forward kinematics example for the Synapse SDK.

``get_link_transforms`` returns ``{frame_name: world_T_link}`` — a 4x4
homogeneous matrix per frame in the kinematic chain at the current joint
configuration.

Universal Robots (UR10e) is used here purely for illustration. It supports
all robots. 

Usage:
    python get_link_transforms.py
"""

from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots
from telekinesis.synapse import utils


def main():
    """Compute every frame's world pose at a chosen joint configuration."""

    # Create the robot (no connect required — runs on the kinematic model)
    robot = universal_robots.UniversalRobotsUR10E()

    # Set a non-trivial joint configuration [deg]
    q = [0, -90, 90, 0, 90, 0]
    robot.set_default_joint_configuration(q=q)

    # Read world transforms for every frame in the kinematic model
    transforms = robot.get_link_transforms()
    logger.info(f"Number of frames at q={q}: {len(transforms)}")

    # Print each frame's pose [x, y, z, rx, ry, rz] (m, deg)
    for name, T in transforms.items():
        pose = utils.transformation_matrix_to_pose(T, rot_type="deg")
        logger.success(f"{name}: pose [m, deg] = {pose}")


if __name__ == "__main__":
    main()
