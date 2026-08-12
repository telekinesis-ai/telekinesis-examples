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
    mesh = fetch_mesh(mesh_url)
    logger.success(f"Loaded mesh with {len(mesh.vertex_positions)} vertices")

    # ===================== Run Skill ==========================================
    filtered_mesh = vitreous.filter_point_cloud_using_cylinder_base_removal(
        mesh=mesh,
        compute_vertex_normals=True,
        distance_threshold=0.005,
    )
    logger.success("Filtered mesh using cylinder base removal")

    # ===================== Visualization  (Optional) ======================
    # Mesh3D has no telekinesis visualize() handler yet, so it is logged directly with Rerun.
    rr.init("filter_point_cloud_using_cylinder_base_removal_example", spawn=True)
    rr.log("/input_mesh", rr.Mesh3D(
        vertex_positions=mesh.vertex_positions,
        triangle_indices=mesh.triangle_indices,
        vertex_colors=mesh.vertex_colors,
        vertex_normals=mesh.vertex_normals,
        albedo_factor=[0.8, 0.8, 0.8, 1.0],
    ))
    rr.log("/filtered_mesh", rr.Mesh3D(
        vertex_positions=filtered_mesh.vertex_positions,
        triangle_indices=filtered_mesh.triangle_indices,
        vertex_colors=filtered_mesh.vertex_colors,
        vertex_normals=filtered_mesh.vertex_normals,
        albedo_factor=[0.8, 0.8, 0.8, 1.0],
    ))


def fetch_mesh(url: str) -> datatypes.Mesh3D:
    """Downloads a mesh from a URL and loads it as a Mesh3D object.

    `Mesh3D` has no `from_url` loader (unlike `PointCloud`), so the mesh is
    decoded via `trimesh` and its arrays are wrapped directly.
    """
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    with tempfile.NamedTemporaryFile(suffix=pathlib.Path(url).suffix, delete=False) as tmp:
        tmp.write(response.content)
        tmp_path = tmp.name
    scene = trimesh.load(tmp_path, force="scene")
    trimesh_mesh = (
        trimesh.util.concatenate(tuple(scene.geometry.values()))
        if isinstance(scene, trimesh.Scene)
        else scene
    )
    pathlib.Path(tmp_path).unlink(missing_ok=True)
    logger.success(f"Loaded mesh from {url}")
    return datatypes.Mesh3D(
        vertex_positions=trimesh_mesh.vertices.astype(np.float32),
        triangle_indices=trimesh_mesh.faces.astype(np.int32),
        vertex_normals=trimesh_mesh.vertex_normals.astype(np.float32)
        if trimesh_mesh.vertex_normals is not None
        else None,
    )


if __name__ == "__main__":
    filter_point_cloud_using_cylinder_base_removal_example()
