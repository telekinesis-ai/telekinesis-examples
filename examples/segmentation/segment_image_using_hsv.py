"""
Demonstrates HSV color space segmentation.

This example:
- Downloads an example image.
- Segments it using HSV color space range.
- Visualizes the result using Rerun.
"""

from loguru import logger
import rerun as rr

from telekinesis import cornea, datatypes

def segment_image_using_hsv_example():
    """Segments an image using HSV color space range."""
    # ===================== Load Image ==========================================
    image_url = "https://assets.telekinesis.ai/examples/v1/images/wires_rgb.png"
    image = datatypes.Image.from_url(url=image_url)
    logger.info(f"Loaded {image} from the URL: {image_url}")

    # ===================== Run Skill ==========================================
    segmented_image = cornea.segment_image_using_hsv(
        image=image, lower_bound=(0, 50, 50), upper_bound=(180, 255, 255)
    )
    logger.success("Segmentation completed.")

    # ===================== Visualization  (Optional) ======================
    rr.init("segment_image_using_hsv_example", spawn=True)
    datatypes.visualize(image, entity_path="/Image/original_image")
    datatypes.visualize(segmented_image, entity_path="/SegmentedImage/segmented_image")


if __name__ == "__main__":
    segment_image_using_hsv_example()
