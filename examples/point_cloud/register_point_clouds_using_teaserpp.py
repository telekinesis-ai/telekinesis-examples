"""
Demonstrates aligning point clouds using TEASER++ with internally derived correspondences.
"""

import numpy as np
from loguru import logger
import rerun as rr

from telekinesis import datatypes, vitreous

import numpy as np
from scipy.spatial import cKDTree


def estimate_point_spacing(point_cloud):
    points = np.asarray(point_cloud.positions)

    tree = cKDTree(points)

    # k=2 because the closest point is the point itself.
    distances, _ = tree.query(points, k=2)

    nearest_neighbor_distances = distances[:, 1]

    return float(np.median(nearest_neighbor_distances))


def register_point_clouds_using_teaserpp_example():
    """
    Aligns point clouds using TEASER++.

    TEASER++ now derives correspondences internally from local geometry by
    estimating normals and computing FPFH features for the source and target
    point clouds.
    """
    # ===================== Load Data ==========================================
    source_point_cloud_url = "https://assets.telekinesis.ai/examples/v1/point_clouds/gusset_model_voxelized.ply"
    target_point_cloud_url = "https://assets.telekinesis.ai/examples/v1/point_clouds/gusset_0_preprocessed_voxelized.ply"
    source_point_cloud = datatypes.PointCloud.from_url(
        url=source_point_cloud_url, use_cache=True
    )
    target_point_cloud = datatypes.PointCloud.from_url(
        url=target_point_cloud_url, use_cache=True
    )
    logger.info(f"Loaded source point cloud: {source_point_cloud}")
    logger.info(f"Loaded target point cloud: {target_point_cloud}")

    source_spacing = estimate_point_spacing(source_point_cloud)
    target_spacing = estimate_point_spacing(target_point_cloud) 

    spacing = max(source_spacing, target_spacing)   

    logger.info(f"Source spacing: {source_spacing}")
    logger.info(f"Target spacing: {target_spacing}")
    logger.info(f"Registration spacing: {spacing}")

    # ===================== Run Skill ==========================================
    result = vitreous.register_point_clouds_using_teaserpp(
        source_point_cloud=source_point_cloud,
        target_point_cloud=target_point_cloud,
        initial_transformation_matrix=np.eye(4),
        normal_radius=spacing * 2.0,
        normal_max_neighbors=30,
        feature_radius=spacing * 5.0,
        feature_max_neighbors=100,
        noise_bound=1 * spacing,
        cbar2=1.0,
        estimate_scaling=False,
        rotation_gnc_factor=1.4,
        rotation_max_iterations=10000,
        rotation_cost_threshold=1e-16,
    )
    correspondence_set = None
    fitness = None
    inlier_rmse = None
    if isinstance(result, tuple):
        (
            transformation_matrix,
            correspondence_set,
            fitness,
            inlier_rmse,
        ) = result
    else:
        transformation_matrix = result

    # ===================== Log ================================================
    logger.success(
        f"Registered {source_point_cloud} to {target_point_cloud} using TEASER++"
    )
    logger.info(f"Final TEASER++ transformation matrix:\n{transformation_matrix}")
    if correspondence_set is not None:
        logger.info(f"Correspondence set shape: {correspondence_set.shape}")
    else:
        logger.info("Correspondence set: not returned by the current SDK/runtime")
    if fitness is not None:
        logger.info(f"Fitness: {fitness}")
    else:
        logger.info("Fitness: not returned by the current SDK/runtime")
    if inlier_rmse is not None:
        logger.info(f"Inlier RMSE: {inlier_rmse}")
    else:
        logger.info("Inlier RMSE: not returned by the current SDK/runtime")
    logger.info(f"Transformation matrix data: {transformation_matrix.data}")
    logger.info(f"Transformation matrix shape: {transformation_matrix.shape}")
    logger.info(f"Transformation matrix ndim: {transformation_matrix.ndim}")
    logger.info(f"Transformation matrix dtype: {transformation_matrix.dtype}")

    # ===================== Visualization  (Optional) ===========================
    aligned_source_point_cloud = vitreous.apply_transform_to_point_cloud(
        point_cloud=source_point_cloud,
        transformation_matrix=transformation_matrix,
    )

    rr.init("register_point_clouds_using_teaserpp_example", spawn=True)
    datatypes.visualize(source_point_cloud, entity_path="/1-before_registration_source")
    datatypes.visualize(target_point_cloud, entity_path="/2-before_registration_target")
    datatypes.visualize(
        aligned_source_point_cloud, entity_path="/3-after_registration_source_aligned"
    )


if __name__ == "__main__":
    register_point_clouds_using_teaserpp_example()
