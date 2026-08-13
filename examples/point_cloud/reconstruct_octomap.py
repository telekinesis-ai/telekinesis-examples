"""
Demonstrates reconstructing occupied and free OctoMap cells from a point cloud.

This example downloads a point cloud, builds an OctoMap, and visualizes the
input cloud alongside its occupied and free cells using Rerun.
"""

from loguru import logger
import rerun as rr

from telekinesis import vitreous, datatypes


def reconstruct_octomap_example():
    """Reconstructs occupied and free OctoMap cells from a point cloud."""
    # ===================== Load Data ==========================================
    point_cloud_url = "https://assets.telekinesis.ai/examples/v1/point_clouds/beer_can_corrupted_normals.ply"
    point_cloud = datatypes.PointCloud.from_url(url=point_cloud_url, use_cache=True)
    logger.success(f"Loaded point cloud with {len(point_cloud)} points")

    # ===================== Run Skill ==========================================
    occupied_cells, free_cells = vitreous.reconstruct_octomap(
        point_cloud=point_cloud,
        resolution=0.01,
        sensor_origin=[0.0, 0.0, 0.0],
    )
    logger.success(
        f"Reconstructed OctoMap with {len(occupied_cells)} occupied and {len(free_cells)} free cells"
    )

    # Access occupied_cells and free_cells data and properties
    occupied_cells_centers = occupied_cells.center
    occupied_cells_volumes = occupied_cells.volume
    free_cells_centers = free_cells.center
    free_cells_volumes = free_cells.volume
    logger.info(f"Occupied cells centers shape: {occupied_cells_centers.shape}, volumes shape: {occupied_cells_volumes.shape}")
    logger.info(f"Free cells centers shape: {free_cells_centers.shape}, volumes shape: {free_cells_volumes.shape}")

    # ===================== Visualization (Optional) ===========================
    rr.init("reconstruct_octomap_example", spawn=True)
    datatypes.visualize(point_cloud, entity_path="/input_point_cloud")
    datatypes.visualize(occupied_cells, entity_path="/output_octomap/occupied_cells")
    datatypes.visualize(free_cells, entity_path="/output_octomap/free_cells")


if __name__ == "__main__":
    reconstruct_octomap_example()
