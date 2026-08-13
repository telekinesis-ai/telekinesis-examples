"""
Demonstrates removing the base faces from a cylindrical mesh.

This example:
- Downloads an example mesh.
- Identifies and removes triangles that form the flat base(s) of a cylinder.
- Visualizes the result using Rerun.
"""

import pathlib
import tempfile

import numpy as np
import requests
import trimesh
from loguru import logger
import rerun as rr

from telekinesis import vitreous, datatypes


def filter_point_cloud_using_cylinder_base_removal_example():
    """
    Removes the base faces from a cylindrical mesh.

    Identifies and removes triangles that form the flat base(s) of a cylinder,
    leaving only the curved side surface.
    """
    # ===================== Load Data ==========================================
    mesh_url = "https://assets.telekinesis.ai/examples/v1/meshes/beer_can.glb"
    mesh = datatypes.Mesh3D.from_url(url=mesh_url, use_cache=True)
    logger.success(f"Loaded mesh with {len(mesh.vertex_positions)} vertices")

    # ===================== Run Skill ==========================================
    filtered_mesh = vitreous.filter_point_cloud_using_cylinder_base_removal(
        mesh=mesh,
        compute_vertex_normals=True,
        distance_threshold=0.005,
    )
    logger.success("Filtered mesh using cylinder base removal")

    # Access the filtered mesh data
    filtered_vertex_positions = filtered_mesh.vertex_positions
    filtered_triangle_indices = filtered_mesh.triangle_indices

    filtered_vertex_normals = filtered_mesh.vertex_normals if filtered_mesh.has_vertex_normals else None
    filtered_vertex_colors = filtered_mesh.vertex_colors if filtered_mesh.has_vertex_colors else None

    logger.info(f"Filtered mesh: {filtered_mesh}")
    logger.info(f"Filtered mesh has {len(filtered_vertex_positions)} vertices "
                f"and {len(filtered_triangle_indices)} triangles")
    logger.info(f"Filtered mesh has vertex normals: {filtered_vertex_normals}, "
                f"vertex colors: {filtered_vertex_colors}")

    # ===================== Visualization  (Optional) ======================
    # Mesh3D has no telekinesis visualize() handler yet, so it is logged directly with Rerun.
    rr.init("filter_point_cloud_using_cylinder_base_removal_example", spawn=True)
    datatypes.visualize(mesh, entity_path="/OriginalMesh/mesh", label="Original Mesh")
    datatypes.visualize(filtered_mesh, entity_path="/FilteredMesh/filtered_mesh", label="Filtered Mesh")


if __name__ == "__main__":
    filter_point_cloud_using_cylinder_base_removal_example()
