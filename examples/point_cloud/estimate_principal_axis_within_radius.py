"""
Demonstrates estimating the principal component axis of a point cloud neighborhood.
"""

from loguru import logger
import rerun as rr

from telekinesis import vitreous, datatypes


def estimate_principal_axis_within_radius_example():
    """
    Estimates the principal component axis of a point cloud neighborhood.

    Uses PCA to find the dominant direction in a local neighborhood around a
    reference point.
    """
    # ===================== Load Data ==========================================
    point_cloud_url = "https://assets.telekinesis.ai/examples/v1/point_clouds/mug_preprocessed.ply"
    point_cloud = datatypes.PointCloud.from_url(url=point_cloud_url, use_cache=True)

    # ===================== Run Skill ==========================================
    # `estimate_principal_axis_within_radius` returns a Vector3D datatype instance.
    local_principal_axis = vitreous.estimate_principal_axis_within_radius(
        point_cloud=point_cloud,
        neighborhood_radius=0.25,
        reference_point=[0.0, 0.0, -0.52],
    )

    # ===================== Log ================================================
    logger.success("Estimated principal axis within radius")
    logger.success(f"Results: {local_principal_axis}")
    logger.info(f"Local principal axis data: {local_principal_axis.data}")
    logger.info(f"Local principal axis shape: {local_principal_axis.shape}")
    logger.info(f"Local principal axis dtype: {local_principal_axis.dtype}")

    # ===================== Visualization  (Optional) ===========================
    rr.init("estimate_principal_axis_within_radius_example", spawn=True)
    datatypes.visualize(point_cloud, entity_path="/1-point_cloud")
    datatypes.visualize(local_principal_axis, entity_path="/2-local_principal_axis", label="Principal Axis")


if __name__ == "__main__":
    estimate_principal_axis_within_radius_example()
