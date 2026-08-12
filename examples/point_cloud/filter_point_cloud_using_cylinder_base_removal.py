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
from loguru import logger
import rerun as rr
from rerun import blueprint as rrb

from datatypes import datatypes, io
from telekinesis import vitreous


def filter_point_cloud_using_cylinder_base_removal_example():
    """
    Removes the base faces from a cylindrical mesh.

    Identifies and removes triangles that form the flat base(s) of a cylinder,
    leaving only the curved side surface.
    """
    # ===================== Load Data ==========================================
    mesh_url = "https://telekinesis-public-assets.s3.us-east-1.amazonaws.com/examples/v1/meshes/beer_can.glb"
    mesh = fetch_mesh(mesh_url)
    logger.success("Loaded mesh")

    # ===================== Run Skill ==========================================
    filtered_mesh = vitreous.filter_point_cloud_using_cylinder_base_removal(
        mesh=mesh,
        compute_vertex_normals=True,
        distance_threshold=0.005
    )

    logger.success("Filtered mesh using cylinder base removal")

    # ===================== Visualization  (Optional) ======================
    visualize(mesh, filtered_mesh)


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


def visualize(mesh, filtered_mesh) -> None:
    """Visualizes the input mesh and the base-removed mesh using Rerun."""
    # Initialize Rerun
    rr.init("filter_point_cloud_using_cylinder_base_removal", spawn=False)
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
    verts = np.asarray(filtered_mesh.vertex_positions)
    bbox_min = verts.min(axis=0)
    bbox_max = verts.max(axis=0)
    mesh_center = 0.5 * (bbox_min + bbox_max)

    look_target = mesh_center
    eye_up = np.array([0.0, 0.0, 0.1])
    offset = eye_up * 2
    position = look_target + offset

    eye_controls = rrb.EyeControls3D(
        kind=rrb.Eye3DKind.Orbital,
        position=position,
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
                name="Input Mesh",
                origin="input_mesh",
                background=background,
                eye_controls=eye_controls,
                line_grid=line_grid,
                spatial_information=spatial_information,
            ),
            rrb.Spatial3DView(
                name="Filtered Mesh",
                origin="filtered_mesh",
                background=background,
                eye_controls=eye_controls,
                line_grid=line_grid,
                spatial_information=spatial_information,
            ),
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

    # Log the output filtered mesh under filtered_mesh
    rr.log("filtered_mesh", rr.Mesh3D(
        vertex_positions=filtered_mesh.vertex_positions,
        triangle_indices=filtered_mesh.triangle_indices,
        vertex_colors=filtered_mesh.vertex_colors,
        vertex_normals=filtered_mesh.vertex_normals,
        albedo_factor=[0.8, 0.8, 0.8, 1.0],
    ))


if __name__ == "__main__":
    filter_point_cloud_using_cylinder_base_removal_example()
