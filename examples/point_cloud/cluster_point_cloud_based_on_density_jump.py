"""
Demonstrates splitting a point cloud into regions based on density discontinuities.

This example:
- Downloads an example point cloud.
- Detects and splits point clouds at locations where point density changes dramatically.
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


def cluster_point_cloud_based_on_density_jump_example():
    """
    Splits a point cloud into regions based on density discontinuities.

    Detects and splits point clouds at locations where point density changes
    dramatically.
    """
    # ===================== Load Data ==========================================
    point_cloud_url = "https://telekinesis-public-assets.s3.us-east-1.amazonaws.com/examples/v1/point_clouds/mug_preprocessed.ply"
    point_cloud = fetch_point_cloud(point_cloud_url)
    logger.success(f"Loaded point cloud with {len(point_cloud.positions)} points")

    # ===================== Run Skill ==========================================
    clusters = vitreous.cluster_point_cloud_based_on_density_jump(
        point_cloud=point_cloud,
        num_nearest_neighbors=5,
        neighborhood_radius=0.05,
        is_point_cloud_linear=False,
        projection_axis=[0.0, 0.0, 1.0],
    )
    logger.success(
        "Split point cloud based on density jump"
    )

    # ===================== Visualization  (Optional) ======================
    visualize(point_cloud, clusters)


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


def visualize(point_cloud, clusters) -> None:
    """Visualizes the input point cloud and the density-jump split clusters using Rerun."""
    # Initialize Rerun
    rr.init("cluster_point_cloud_based_on_density_jump", spawn=False)
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
    overview_position = np.array([-2.5, 2.5, 5.])
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
            rrb.Spatial3DView(name="Output Point Cloud", origin="splitted_clusters",
                              eye_controls=eye_controls,
                              background=background,
                              spatial_information=spatial_information,
                              line_grid=line_grid),
        )
    ))

    rr.log("input_point_cloud", rr.Points3D(positions=point_cloud.positions,
           colors=point_cloud.colors))

    # Log each output point cloud under its own path
    for i, cluster in enumerate(clusters.to_list()):
            rr.log(f"splitted_clusters/cluster_{i+1}", rr.Points3D(positions=cluster.positions,
                   colors=cluster.colors))


if __name__ == "__main__":
    cluster_point_cloud_based_on_density_jump_example()
