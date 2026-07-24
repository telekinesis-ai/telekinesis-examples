"""
Demonstrates computing the convex hull mesh enclosing a point cloud.

This example:
- Downloads an example point cloud.
- Computes the smallest convex shape that contains all points.
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


def reconstruct_mesh_using_convex_hull_example():
    """
    Computes the convex hull mesh enclosing a point cloud.

    Creates the smallest convex shape that contains all points.
    """
    # ===================== Load Data ==========================================
    point_cloud_url = "https://telekinesis-public-assets.s3.us-east-1.amazonaws.com/examples/v1/point_clouds/beer_can_corrupted_normals.ply"
    point_cloud = fetch_point_cloud(point_cloud_url)
    logger.success(f"Loaded point cloud with {len(point_cloud.positions)} points")

    # ===================== Run Skill ==========================================
    result_mesh = vitreous.reconstruct_mesh_using_convex_hull(
        joggle_inputs=False,
        point_cloud=point_cloud
    )
    logger.success(
        f"Reconstructed convex hull mesh from {len(point_cloud.positions)} points"
    )

    # ===================== Visualization  (Optional) ======================
    visualize(point_cloud, result_mesh)


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


def visualize(point_cloud, result_mesh) -> None:
    """Visualizes the input point cloud and the reconstructed convex hull mesh using Rerun."""
    # Initialize Rerun
    rr.init("reconstruct_mesh_using_convex_hull", spawn=False)
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
    overview_position = np.array([0.28564838, -0.01504822, 0.3774886])
    look_target = np.array([-0.00074809, -0.00017695, 0.06533939])
    eye_up = np.array([0.51810353, -0.85454376, -0.03638268])

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
                name="Convex Hull Reconstructed Mesh",
                origin="convex_hull_mesh",
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
    rr.log("convex_hull_mesh", rr.Mesh3D(
        vertex_positions=result_mesh.vertex_positions,
        triangle_indices=result_mesh.triangle_indices,
        vertex_normals=result_mesh.vertex_normals if hasattr(result_mesh, 'vertex_normals') else None,
        albedo_factor=[0.8, 0.8, 0.8, 1.0],
    ))


if __name__ == "__main__":
    reconstruct_mesh_using_convex_hull_example()
