"""
Read per-link visual mesh data.

Supports Universal Robots (UR), Epson, and virtual/sim.

Usage:
    python get_visual_meshes_data.py
"""

from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots


def main():
    """Read per-link visual mesh data and log a summary per link."""

    #===================== Create Robot ==========================================
    robot = universal_robots.UniversalRobotsUR10E(name='UR10e')

    # ==================== Run Skill ============================================
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
