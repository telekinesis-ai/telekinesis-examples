"""
Demonstrates creating a parametric cylinder mesh.

This example:
- Generates a cylinder with specified radius, height, and resolution.
- Visualizes the result using Rerun.
"""

import numpy as np
from loguru import logger
import rerun as rr
from rerun import blueprint as rrb

from telekinesis import vitreous


def create_cylinder_mesh_example():
    """
    Creates a parametric cylinder mesh.

    Generates a cylinder with specified radius, height, and resolution.
    """
    # ===================== Run Skill ==========================================
    cylinder_mesh = vitreous.create_cylinder_mesh(
        radius=0.01,
        height=0.02,
        radial_resolution=20,
        height_resolution=4,
        retain_base=False,
        vertex_tolerance=1e-6,
        transformation_matrix=np.eye(4, dtype=np.float32),
        compute_vertex_normals=True,
    )
    logger.success("Created cylinder mesh")

    # ===================== Visualization  (Optional) ======================
    visualize(cylinder_mesh)


def visualize(cylinder_mesh) -> None:
    """Visualizes the generated cylinder mesh using Rerun."""
    # Initialize Rerun
    rr.init("create_cylinder_mesh", spawn=False)
    try:
        rr.connect()
    except Exception:
        rr.spawn()

    # Setup additional rerun settings
    line_grid = rrb.LineGrid3D(visible=True)
    spatial_information = rrb.SpatialInformation(
        target_frame="tf#/",
        show_axes=False,
        show_bounding_box=False,
    )
    background = rrb.Background(color=(255, 255, 255))

    # Setup camera view
    overview_position = np.array([0.02670, 0.04005, 0.02670])
    look_target = np.array([0, 0, 0])
    eye_up = np.array([0., 0., 1.])

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
    rr.send_blueprint(
        rrb.Blueprint(
            rrb.Spatial3DView(
                name="Cylinder Mesh",
                origin="cylinder_mesh",
                background=background,
                eye_controls=eye_controls,
                line_grid=line_grid,
                spatial_information=spatial_information
            ),
        )
    )

    rr.log("cylinder_mesh", rr.Mesh3D(
        vertex_positions=cylinder_mesh.vertex_positions,
        triangle_indices=cylinder_mesh.triangle_indices,
        vertex_normals=cylinder_mesh.vertex_normals,
        albedo_factor=[0.8, 0.8, 0.8, 1.0],
    ))


if __name__ == "__main__":
    create_cylinder_mesh_example()
