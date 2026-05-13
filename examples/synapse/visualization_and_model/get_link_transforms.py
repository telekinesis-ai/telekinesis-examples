"""
Read per-link world transforms for the Synapse SDK.

``get_link_transforms`` returns ``{frame_name: world_T_link}`` — a 4x4
homogeneous matrix per frame in the kinematic model. Useful for
visualization, attaching coordinate axes, or computing relative transforms
between arbitrary frames without calling ``forward_kinematics`` per frame.

Universal Robots (UR10e) is used here purely for illustration.
This example runs purely on the kinematic model and does not connect to
hardware — no ``--ip`` is required.

Usage:
    python get_link_transforms.py
"""

from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots
from telekinesis.synapse import utils


def main():
    """Read every link's world transform at the current joint configuration."""

    # Create the robot (no connect required — runs on the kinematic model)
    robot = universal_robots.UniversalRobotsUR10E()

    # Read world transforms for every frame in the kinematic model
    transforms = robot.get_link_transforms()
    logger.info(f"Number of frames: {len(transforms)}")

    # Convert each 4x4 transformation matrix to a pose [x, y, z, rx, ry, rz] (m, deg)
    for name, T in transforms.items():
        pose = utils.transformation_matrix_to_pose(T, rot_type="deg")
        logger.success(f"{name}: pose [m, deg] = {pose}")


if __name__ == "__main__":
    main()
