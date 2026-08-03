"""
Demonstrates creating a torus (donut shape) mesh.

This example:
- Generates a parametric torus with specified major/minor radii and resolution.
- Visualizes the result using Rerun.
"""

import numpy as np
from loguru import logger
import rerun as rr
from rerun import blueprint as rrb

from telekinesis import vitreous


def create_torus_mesh_example():
    """
    Creates a torus (donut shape) mesh.

    Generates a parametric torus with specified major/minor radii and resolution.
    """
    # ===================== Run Skill ==========================================
    torus_mesh = vitreous.create_torus_mesh(
        transformation_matrix=np.eye(4, dtype=np.float32),
        torus_radius=0.01,
        tube_radius=0.005,
        radial_resolution=20,
        tubular_resolution=10,
        compute_vertex_normals=True,
    )
    logger.success(
        "Created torus mesh"
    )

    # ===================== Visualization  (Optional) ======================
    visualize(torus_mesh)


def visualize(torus_mesh) -> None:
    """Visualizes the generated torus mesh using Rerun."""
    # Initialize Rerun
    rr.init("create_torus_mesh", spawn=False)
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
                name="Torus Mesh",
                origin="torus_mesh",
                background=background,
                eye_controls=eye_controls,
                line_grid=line_grid,
                spatial_information=spatial_information
            ),
        )
    )

    rr.log("torus_mesh", rr.Mesh3D(
        vertex_positions=torus_mesh.vertex_positions,
        triangle_indices=torus_mesh.triangle_indices,
        vertex_normals=torus_mesh.vertex_normals,
        albedo_factor=[0.8, 0.8, 0.8, 1.0],
    ))


if __name__ == "__main__":
    create_torus_mesh_example()
