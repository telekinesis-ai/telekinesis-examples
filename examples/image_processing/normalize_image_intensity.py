"""Demonstrates normalize_image_intensity operation."""

from loguru import logger
import rerun as rr

from telekinesis import pupil, datatypes


def normalize_image_intensity_example():
    """Applies normalize_image_intensity operation."""
    # ===================== Load Image ==========================================
    # 16-bit bin-picking depth map (0.1 mm units); the parts span only ~4 cm.
    # Example data: XYZ-IBD (CC BY-NC-SA 4.0), modified. Credits and licences: THIRD_PARTY_DATA.md
    image_url = "https://assets.telekinesis.ai/examples/v1/depth_images/xyzibd_flat_bars_depth_16bit.png"
    depth_image = datatypes.DepthImage.from_url(image_url, depth_scale=0.0001)
    image = datatypes.Image(depth_image.to_numpy())

    # ===================== Run Skill ==========================================
    filtered_image = pupil.normalize_image_intensity(
        image=image,
        alpha=0.0,
        beta=255.0,
        normalization_method="minmax",
        output_format="8bit",
    )

    # ===================== Log ================================================
    logger.success(f"Applied normalize_image_intensity on {image}")
    logger.success(f"Result: {filtered_image}")

    # ===================== Visualization  (Optional) ======================
    rr.init("normalize_image_intensity_example", spawn=True)
    datatypes.visualize(image, entity_path="1-original")
    datatypes.visualize(filtered_image, entity_path="2-normalized")

if __name__ == "__main__":
    normalize_image_intensity_example()
