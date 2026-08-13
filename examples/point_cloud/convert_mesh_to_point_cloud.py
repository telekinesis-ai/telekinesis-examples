"""
Demonstrates converting a triangle mesh to a point cloud via surface sampling.

This example:
- Downloads an example mesh.
- Samples points on the mesh surface using uniform or Poisson disk sampling.
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


def convert_mesh_to_point_cloud_example():
    """
    Converts a triangle mesh to a point cloud via surface sampling.

    Samples points on the mesh surface using uniform or Poisson disk sampling.
    """
    # ===================== Load Data ==========================================
    mesh_url = "https://assets.telekinesis.ai/examples/v1/meshes/gear_box.glb"
    mesh = datatypes.Mesh3D.from_url(url=mesh_url, use_cache=True)
    logger.success(f"Loaded mesh with {len(mesh.vertex_positions)} vertices")

    # ===================== Run Skill ==========================================
    point_cloud = vitreous.convert_mesh_to_point_cloud(
        mesh=mesh,
        num_points=10000,
        sampling_method="poisson_disk",
        initial_sampling_factor=5,
        initial_point_cloud=None,
        use_triangle_normal=False,
    )
    logger.success(f"Converted mesh with {len(mesh.vertex_positions)} vertices to point cloud with {len(point_cloud)} points")

    # ===================== Visualization  (Optional) ======================
    rr.init("convert_mesh_to_point_cloud_example", spawn=True)
    # Mesh3D has no telekinesis visualize() handler yet, so it is logged directly with Rerun.
    rr.log("/input_mesh", rr.Mesh3D(
        vertex_positions=mesh.vertex_positions,
        triangle_indices=mesh.triangle_indices,
        vertex_colors=mesh.vertex_colors,
        vertex_normals=mesh.vertex_normals,
        albedo_factor=[0.8, 0.8, 0.8, 1.0],
    ))
    datatypes.visualize(point_cloud, entity_path="/output_point_cloud")


if __name__ == "__main__":
    convert_mesh_to_point_cloud_example()
