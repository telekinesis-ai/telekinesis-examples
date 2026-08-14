"""
Demonstrates cropping an image using multiple bounding boxes.

This example:
- Downloads an example image.
- Crops regions using multiple bounding boxes.
- Visualizes the result using Rerun.
"""

from loguru import logger
import rerun as rr

from telekinesis import pupil, datatypes


def crop_image_using_bounding_boxes_example():
    """Crops image using bounding boxes."""
    # ===================== Load Image ==========================================
    image_url = "https://assets.telekinesis.ai/examples/v1/images/driver_screw.png"
    image = datatypes.Image.from_url(image_url)

    # ===================== Run Skill ==========================================
    # Define bounding boxes in the format [x, y, width, height]
    bounding_boxes = [
        [65, 235, 330, 240],
        [370, 35, 330, 155],
        [445, 210, 85, 300],
    ]

    cropped_images = pupil.crop_image_using_bounding_boxes(
        image=image,
        bounding_boxes=bounding_boxes,
        retain_coordinates=True,
    )

    cropped_image_list = cropped_images.to_list()
    logger.success("Cropped {} regions", len(cropped_image_list))

    # ===================== Visualization  (Optional) ======================
    rr.init("crop_image_using_bounding_boxes_example", spawn=True)
    datatypes.visualize(image, entity_path="1-Original")
    for i, cropped_image in enumerate(cropped_image_list):
        datatypes.visualize(cropped_image, entity_path=f"{i + 2}-Crop {i + 1}")

if __name__ == "__main__":
    crop_image_using_bounding_boxes_example()
