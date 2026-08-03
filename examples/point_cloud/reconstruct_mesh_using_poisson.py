"""
Demonstrates reconstructing a watertight mesh from an oriented point cloud using Poisson surface reconstruction.

This example:
- Downloads an example point cloud.
- Solves a Poisson equation to fit a smooth, closed surface through points with normals.
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


def reconstruct_mesh_using_poisson_example():
    """
    Reconstructs a watertight mesh from an oriented point cloud using Poisson surface reconstruction.

    Solves a Poisson equation to fit a smooth surface through points with normals.
    Produces closed, manifold meshes. Requires point cloud normals.
    """
    # ===================== Load Data ==========================================
    point_cloud_url = "https://telekinesis-public-assets.s3.us-east-1.amazonaws.com/examples/v1/point_clouds/industrial_part_7_normals.ply"
    point_cloud = fetch_point_cloud(point_cloud_url)
    logger.success(f"Loaded point cloud with {len(point_cloud.positions)} points")

    # ===================== Run Skill ==========================================
    reconstructed_mesh = vitreous.reconstruct_mesh_using_poisson(
        octree_depth=7, octree_width=0, scale_factor=1.1,
        point_cloud=point_cloud,
    )
    logger.success(
        f"Reconstructed mesh from {len(point_cloud.positions)} points using Poisson"
    )

    # ===================== Visualization  (Optional) ======================
    visualize(point_cloud, reconstructed_mesh)


def fetch_point_cloud(url: str) -> datatypes.Points3D:
    """Downloads a PLY point cloud from a URL and loads it as a Points3D object."""
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    with tempfile.NamedTemporaryFile(suffix=pathlib.Path(url).suffix, delete=False) as tmp:
        tmp.write(response.content)
        tmp_path = tmp.name
    point_cloud = io.load_point_cloud(filepath=tmp_path)
    pathlib.Path(tmp_path).unlink(missing_ok=True)
    logger.success(f"Loaded point cloud from {url}")
    return point_cloud


def visualize(point_cloud, reconstructed_mesh) -> None:
    """Visualizes the input point cloud and the Poisson-reconstructed mesh using Rerun."""
    # Initialize Rerun
    rr.init("reconstruct_mesh_using_poisson", spawn=False)
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
    overview_position = np.array([0.41282293, -0.59672095, -0.14365284])
    look_target = np.array([-0.18742625, 0.00011380, 0.00008558])
    eye_up = np.array([0.00068377, -0.97230825, -0.23370111])

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
            rrb.Spatial3DView(
                name="Input Point Cloud",
                origin="input_point_cloud",
                background=background,
                eye_controls=eye_controls,
                line_grid=line_grid,
                spatial_information=spatial_information,
            ),
            rrb.Spatial3DView(
                name="Poisson Reconstructed Mesh",
                origin="poisson_mesh",
                background=background,
                eye_controls=eye_controls,
                line_grid=line_grid,
                spatial_information=spatial_information,
            ),
        )
    ))

    # Log the input point cloud
    rr.log("input_point_cloud", rr.Points3D(
        positions=point_cloud.positions,
        colors=point_cloud.colors if point_cloud.colors is not None else None,
    ))

    # Log the output mesh
    rr.log("poisson_mesh", rr.Mesh3D(
        vertex_positions=reconstructed_mesh.vertex_positions,
        triangle_indices=reconstructed_mesh.triangle_indices,
        vertex_normals=reconstructed_mesh.vertex_normals if hasattr(reconstructed_mesh, 'vertex_normals') else None,
        albedo_factor=[0.8, 0.8, 0.8, 1.0],
    ))


if __name__ == "__main__":
    reconstruct_mesh_using_poisson_example()
