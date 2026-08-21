"""
Demonstrates filtering a structured point cloud using a 2D binary mask, keeping only points where the mask is True.
"""

from loguru import logger
import rerun as rr
import rerun.blueprint as rrb

from telekinesis import vitreous, datatypes, cornea


def filter_point_cloud_using_mask_example():
    """
    Filters a structured point cloud using a 2D binary mask.

    Applies a 2D image mask to an organized point cloud, keeping only points
    where the corresponding pixel is True.
    """
    # ===================== Load Data ==========================================
    point_cloud_url = (
        "https://assets.telekinesis.ai/examples/v1/point_clouds/can_vertical_6_raw.ply"
    )
    point_cloud = datatypes.PointCloud.from_url(url=point_cloud_url, use_cache=True)

    mask_url = (
        "https://assets.telekinesis.ai/examples/v1/images/can_vertical_6_mask.png"
    )
    mask_image = datatypes.Image.from_url(url=mask_url)
    mask = cornea.segment_image_using_threshold(image=mask_image, min_value=127)

    # ===================== Run Skill ==========================================
    result_point_cloud = vitreous.filter_point_cloud_using_mask(
        point_cloud=point_cloud,
        mask=mask,
    )

    # ===================== Log ================================================
    logger.success(f"Filtered {point_cloud} using mask")
    logger.success(f"Results: {result_point_cloud}")
    logger.info(
        f"Result point cloud positions shape: {result_point_cloud.positions.shape}"
    )
    logger.info(
        f"Result point cloud has normals shape: "
        f"{result_point_cloud.normals.shape if result_point_cloud.has_normals else None}"
    )
    logger.info(
        f"Result point cloud has colors shape: "
        f"{result_point_cloud.colors.shape if result_point_cloud.has_colors else None}"
    )

    # ===================== Visualization  (Optional) ===========================
    rr.init("filter_point_cloud_using_mask_example", spawn=True)
    rr.send_blueprint(
        rrb.Blueprint(
            rrb.Horizontal(
                rrb.Spatial3DView(
                    name="Input Point Cloud",
                    origin="/1-input_point_cloud",
                ),
                rrb.Spatial2DView(
                    name="Binary Mask",
                    origin="/2-binary_mask",
                ),
                rrb.Spatial3DView(
                    name="Masked Point Cloud",
                    origin="/3-masked_point_cloud",
                ),
            )
        )
    )
    datatypes.visualize(point_cloud, entity_path="/1-input_point_cloud")
    datatypes.visualize(mask, entity_path="/2-binary_mask")
    datatypes.visualize(result_point_cloud, entity_path="/3-masked_point_cloud")


if __name__ == "__main__":
    filter_point_cloud_using_mask_example()
