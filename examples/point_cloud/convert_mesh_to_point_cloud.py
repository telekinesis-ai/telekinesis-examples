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
    mesh = fetch_mesh(mesh_url)
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
    convert_mesh_to_point_cloud_example()
