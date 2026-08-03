"""
Demonstrates estimating the principal component axis of a point cloud neighborhood.

This example:
- Downloads an example point cloud.
- Uses PCA to find the dominant direction in a local neighborhood around a reference point.
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


def estimate_principal_axis_within_radius_example():
    """
    Estimates the principal component axis of a point cloud neighborhood.

    Uses PCA to find the dominant direction in a local neighborhood around a
    reference point.
    """
    # ===================== Load Data ==========================================
    point_cloud_url = "https://telekinesis-public-assets.s3.us-east-1.amazonaws.com/examples/v1/point_clouds/mug_preprocessed.ply"
    point_cloud = fetch_point_cloud(point_cloud_url)
    logger.success(f"Loaded point cloud with {len(point_cloud.positions)} points")

    reference_point = np.array([0., 0., -0.52], dtype=np.float32)
    neighborhood_radius = .25

    # ===================== Run Skill ==========================================
    local_principal_axis = vitreous.estimate_principal_axis_within_radius(
        point_cloud=point_cloud,
        neighborhood_radius=neighborhood_radius,
        reference_point=reference_point,
    )
    logger.success(
        "Estimated principal axis within radius"
    )

    # ===================== Visualization  (Optional) ======================
    visualize(point_cloud, local_principal_axis, reference_point, neighborhood_radius)


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


def visualize(point_cloud, local_principal_axis, reference_point, neighborhood_radius) -> None:
    """Visualizes the point cloud, reference neighborhood, and principal axis using Rerun."""
    # Initialize Rerun
    rr.init("estimate_principal_axis_within_radius", spawn=False)
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
    overview_position = np.array([-1.25, 1.25, 2.5])
    look_target = np.array([ 9.22595333e-18, -1.26803568e-15, -3.16024984e-17])
    eye_up = np.array([0., 1., 0.])

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
            rrb.Spatial3DView(name="Input Point Cloud",
                              origin="input_point_cloud",
                              eye_controls=eye_controls,
                              background=background,
                              spatial_information=spatial_information,
                              line_grid=line_grid),
        )
    ))

    rr.log("input_point_cloud", rr.Points3D(np.asarray(point_cloud.positions),
           colors=(np.asarray(point_cloud.colors))))
    # Visualize reference point (green sphere)
    rr.log("input_point_cloud/reference_point", rr.Points3D(
        np.array([reference_point]),
        colors=np.array([[255, 0, 0]]),
        radii=[.03]
    ))

    # Visualize neighborhood sphere boundary (yellow)
    u = np.linspace(0, 2 * np.pi, 60)
    v = np.linspace(0, np.pi, 40)
    x = neighborhood_radius * np.outer(np.cos(u), np.sin(v)) + reference_point[0]
    y = neighborhood_radius * np.outer(np.sin(u), np.sin(v)) + reference_point[1]
    z = neighborhood_radius * np.outer(np.ones(np.size(u)), np.cos(v)) + reference_point[2]
    sphere_points = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=1)
    rr.log("input_point_cloud/neighborhood_sphere", rr.Points3D(
        sphere_points,
        colors=[[255, 255, 0]] * len(sphere_points),
        # radii=[1.0] * len(sphere_points)
    ))

    # Visualize principal axis arrow (red)
    rr.log("input_point_cloud/arrow", rr.Arrows3D(
        origins=np.array(reference_point),
        vectors=np.array(local_principal_axis),
        colors=np.array([[255, 0, 0]]),
        radii=0.1
    ))


if __name__ == "__main__":
    estimate_principal_axis_within_radius_example()
