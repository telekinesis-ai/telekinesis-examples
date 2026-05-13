"""
Read per-link visual mesh data (vertices/triangles/normals) for the Synapse SDK.

``get_visual_meshes_data`` parses the robot's URDF and returns raw
vertex/triangle/normal arrays ready for visualizers.

Universal Robots (UR10e) is used here purely for illustration. It supports all robots.

Usage:
    python get_visual_meshes_data.py
"""

from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots


def main():
    """Read per-link visual mesh data and log a summary per link."""

    # Create the robot (no connect required — runs on the kinematic model)
    robot = universal_robots.UniversalRobotsUR10E()

    # Read per-link visual mesh data
    meshes = robot.get_visual_meshes_data()
    logger.info(f"Number of links: {len(meshes)}")

    # Log a shape summary per link (vertices / triangles / colors)
    for link_name, mesh in meshes.items():
        if mesh["vertices"] is None:
            logger.warning(f"{link_name}: no visual mesh")
            continue
        n_vertices = mesh["vertices"].shape[0]
        n_triangles = mesh["triangles"].shape[0]
        has_colors = mesh["vertex_colors"] is not None
        logger.success(
            f"{link_name}: vertices={n_vertices}, triangles={n_triangles}, "
            f"vertex_colors={has_colors}, mesh_origin={mesh['mesh_origin']}"
        )


if __name__ == "__main__":
    main()
