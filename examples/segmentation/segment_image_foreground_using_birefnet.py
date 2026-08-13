"""
Demonstrates foreground segmentation using BiRefNet.

This example:
- Downloads an example image.
- Segments the foreground using a pretrained BiRefNet model.
- Visualizes the result using Rerun.
"""

from loguru import logger
import rerun as rr
from telekinesis import cornea, datatypes

def segment_image_foreground_using_birefnet_example():
    """Segments the foreground from the background using BiRefNet."""
    # ===================== Load Image ==========================================
    image_url = "https://assets.telekinesis.ai/examples/v1/images/screws_standing.jpg"
    image = datatypes.Image.from_url(url=image_url)
    logger.info(f"Loaded {image} from the URL: {image_url}")

    # ===================== Run Skill ==========================================
    segmentation_image = cornea.segment_image_foreground_using_birefnet(
        image=image, mask_threshold=0
    )
    logger.success("Segmentation completed.")

    # ===================== Visualization  (Optional) ======================
    rr.init("segment_image_foreground_using_birefnet_example", spawn=True)
    datatypes.visualize(image, segmentation_image)


if __name__ == "__main__":
    segment_image_foreground_using_birefnet_example()
