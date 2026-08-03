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
from loguru import logger
import rerun as rr
from rerun import blueprint as rrb

from datatypes import datatypes, io
from telekinesis import vitreous


def convert_mesh_to_point_cloud_example():
    """
    Converts a triangle mesh to a point cloud via surface sampling.

    Samples points on the mesh surface using uniform or Poisson disk sampling.
    """
    # ===================== Load Data ==========================================
    mesh_url = "https://telekinesis-public-assets.s3.us-east-1.amazonaws.com/examples/v1/meshes/gear_box.glb"
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
    logger.success(
        f"Converted mesh with {len(mesh.vertex_positions)} vertices to point cloud"
    )

    # ===================== Visualization  (Optional) ======================
    visualize(mesh, point_cloud)


def fetch_mesh(url: str) -> datatypes.Mesh3D:
    """Downloads a mesh from a URL and loads it as a Mesh3D object."""
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    with tempfile.NamedTemporaryFile(suffix=pathlib.Path(url).suffix, delete=False) as tmp:
        tmp.write(response.content)
        tmp_path = tmp.name
    mesh = io.load_mesh(filepath=tmp_path)
    pathlib.Path(tmp_path).unlink(missing_ok=True)
    logger.success(f"Loaded mesh from {url}")
    return mesh


def visualize(mesh, point_cloud) -> None:
    """Visualizes the input mesh and the sampled output point cloud using Rerun."""
    # Initialize Rerun
    rr.init("convert_mesh_to_point_cloud", spawn=False)
    try:
        rr.connect()
    except Exception:
        rr.spawn()

    # Setup additional rerun settings
    line_grid = rrb.LineGrid3D(visible=False)
    spatial_information = rrb.SpatialInformation(
        target_frame="tf#/",
        show_axes=False,
        show_bounding_box=False,
    )
    background = rrb.Background(color=(255, 255, 255))

    # Setup camera view
    overview_position = np.array([399.73988139, 599.90846721, 400.29698451])
    look_target = np.array([0.0867062, 0.03051093, -0.09899484])
    eye_up = np.array([0.0, 0.0, 1.0])

    eye_controls = rrb.EyeControls3D(
        kind=rrb.Eye3DKind.Orbital,
        position=overview_position,
        look_target=look_target,
        eye_up=eye_up,
        spin_speed=0.5,
        speed=0.0,
        tracking_entity=None,
    )

    # Send blueprint
    rr.send_blueprint(rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(name="Input Mesh", origin="input_mesh",
                              background=background,
                              eye_controls=eye_controls,
                              line_grid=line_grid,
                              spatial_information=spatial_information),
            rrb.Spatial3DView(name="Output Point Cloud",
                              origin="output_point_cloud",
                              background=background,
                              eye_controls=eye_controls,
                              line_grid=line_grid,
                              spatial_information=spatial_information),
        )
    ))

    # Log the input mesh under input_mesh
    rr.log("input_mesh", rr.Mesh3D(
        vertex_positions=mesh.vertex_positions,
        triangle_indices=mesh.triangle_indices,
        vertex_colors=mesh.vertex_colors,
        vertex_normals=mesh.vertex_normals,
        albedo_factor=[0.8, 0.8, 0.8, 1.0],
    ))

    # Log the output point cloud under output_point_cloud
    rr.log("output_point_cloud", rr.Points3D(positions=point_cloud.positions,
                                             colors=point_cloud.colors))


if __name__ == "__main__":
    convert_mesh_to_point_cloud_example()
