"""
Demonstrates reconstructing occupied and free OctoMap cells from a point cloud.
"""

from loguru import logger
import rerun as rr

from telekinesis import vitreous, datatypes


def reconstruct_octomap_example():
    """
    Reconstructs occupied and free OctoMap cells from a point cloud.

    Builds an OctoMap from the point cloud and returns its occupied and free cells.
    """
    # ===================== Load Data ==========================================
    point_cloud_url = "https://assets.telekinesis.ai/examples/v1/point_clouds/beer_can_corrupted_normals.ply"
    point_cloud = datatypes.PointCloud.from_url(url=point_cloud_url, use_cache=True)

    # ===================== Run Skill ==========================================
    occupied_cells, free_cells = vitreous.reconstruct_octomap(
        point_cloud=point_cloud,
        resolution=0.01,
        sensor_origin=[0.0, 0.0, 0.0],
    )

    # ===================== Log ================================================
    logger.success(
        f"Reconstructed OctoMap with {len(occupied_cells)} occupied and {len(free_cells)} free cells"
    )
    logger.success(f"Results: {occupied_cells}, {free_cells}")
    logger.info(f"Occupied cells centers shape: {occupied_cells.center.shape}, volumes shape: {occupied_cells.volume.shape}")
    logger.info(f"Free cells centers shape: {free_cells.center.shape}, volumes shape: {free_cells.volume.shape}")

    # ===================== Visualization  (Optional) ===========================
    rr.init("reconstruct_octomap_example", spawn=True)
    datatypes.visualize(point_cloud, entity_path="/1-input_point_cloud")
    datatypes.visualize(occupied_cells, entity_path="/2-output_octomap/occupied_cells")
    datatypes.visualize(free_cells, entity_path="/3-output_octomap/free_cells")


if __name__ == "__main__":
    reconstruct_octomap_example()
