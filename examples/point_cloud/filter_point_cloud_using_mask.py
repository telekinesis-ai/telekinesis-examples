"""
Demonstrates filtering a structured point cloud using a 2D binary mask.

This example:
- Downloads an example point cloud and a binary mask image.
- Applies the 2D image mask to the organized point cloud, keeping only points where the mask is True.
- Visualizes the result using Rerun.
"""

from loguru import logger
import rerun as rr
import rerun.blueprint as rrb

from telekinesis import vitreous, datatypes


def filter_point_cloud_using_mask_example():
    """
    Filters a structured point cloud using a 2D binary mask.

    Applies a 2D image mask to an organized point cloud, keeping only points
    where the corresponding pixel is True.
    """
    # ===================== Load Data ==========================================
    point_cloud_url = "https://assets.telekinesis.ai/examples/v1/point_clouds/can_vertical_6_raw.ply"
    point_cloud = datatypes.PointCloud.from_url(url=point_cloud_url, use_cache=True)

    mask_url = "https://assets.telekinesis.ai/examples/v1/images/can_vertical_6_mask.png"
    mask = datatypes.Image.from_url(url=mask_url).to_binary_mask(threshold=127)
    logger.success(f"Loaded point cloud with {len(point_cloud)} points and mask")

    # ===================== Run Skill ==========================================
    result_point_cloud = vitreous.filter_point_cloud_using_mask(
        point_cloud=point_cloud,
        mask=mask,
    )
    logger.success(f"Filtered point cloud to {len(result_point_cloud)} points using mask")

    # Access result_point_cloud data and properties
    result_point_cloud_positions = result_point_cloud.positions
    result_point_cloud_normals = result_point_cloud.normals
    result_point_cloud_colors = result_point_cloud.colors
    logger.info(f"Result point cloud positions shape: {result_point_cloud_positions.shape}")
    logger.info(f"Result point cloud has normals: {result_point_cloud_normals is not None}")
    logger.info(f"Result point cloud has colors: {result_point_cloud_colors is not None}")

    # ===================== Visualization  (Optional) ======================
    rr.init("filter_point_cloud_using_mask_example", spawn=True)
    rr.send_blueprint(
        rrb.Blueprint(
            rrb.Horizontal(
                rrb.Spatial3DView(
                    name="Input Point Cloud",
                    origin="/input_point_cloud",
                ),
                rrb.Spatial2DView(
                    name="Binary Mask",
                    origin="/binary_mask",
                ),
                rrb.Spatial3DView(
                    name="Masked Point Cloud",
                    origin="/masked_point_cloud",
                ),
            )
        )
    )
    datatypes.visualize(point_cloud, entity_path="/input_point_cloud")
    datatypes.visualize(mask, entity_path="/binary_mask")
    datatypes.visualize(result_point_cloud, entity_path="/masked_point_cloud")


if __name__ == "__main__":
    filter_point_cloud_using_mask_example()
