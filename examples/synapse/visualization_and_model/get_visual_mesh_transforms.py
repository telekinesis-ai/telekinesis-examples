"""
Read per-link visual mesh world transforms for the Synapse SDK.

``get_visual_mesh_transforms`` returns ``{link_name: world_T_visual_mesh}``
— a 4x4 homogeneous matrix per link, composed from the link frame and the
URDF ``<visual><origin>`` offset. Links without a usable visual mesh are
omitted. Useful for rendering: each transform is the exact pose at which
the corresponding mesh should be drawn.

Universal Robots (UR10e) is used here purely for illustration.
This example runs purely on the kinematic model and does not connect to
hardware — no ``--ip`` is required.

Usage:
    python get_visual_mesh_transforms.py
"""

from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots
from telekinesis.synapse import utils


def main():
    """Read every link's visual-mesh world transform at the current joint configuration."""

    # Create the robot (no connect required — runs on the kinematic model)
    robot = universal_robots.UniversalRobotsUR10E()

    # Read world transforms for every link's visual mesh
    transforms = robot.get_visual_mesh_transforms()
    logger.info(f"Number of links with visual meshes: {len(transforms)}")

    # Convert each 4x4 transformation matrix to a pose [x, y, z, rx, ry, rz] (m, deg)
    for name, T in transforms.items():
        pose = utils.transformation_matrix_to_pose(T, rot_type="deg")
        logger.success(f"{name}: pose [m, deg] = {pose}")


if __name__ == "__main__":
    main()
