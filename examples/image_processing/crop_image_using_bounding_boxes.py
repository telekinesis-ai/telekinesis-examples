"""Demonstrates cropping an image using multiple bounding boxes."""

from loguru import logger
import rerun as rr

from telekinesis import pupil, datatypes


def crop_image_using_bounding_boxes_example():
    """Crops image using bounding boxes."""
    # ===================== Load Image ==========================================
    # Example data: VisA (CC BY 4.0), modified. Credits and licences: THIRD_PARTY_DATA.md
    image_url = "https://assets.telekinesis.ai/examples/v1/images/visa_pcb_charger_module.jpg"
    image = datatypes.Image.from_url(image_url)

    # ===================== Run Skill ==========================================
    # Define bounding boxes in the format [x, y, width, height]
    bounding_boxes = [
        [247, 330, 205, 340],
        [702, 465, 185, 270],
        [732, 238, 155, 225],
        [892, 235, 135, 545],
    ]

    # Returns the tight crops and each crop's top-left [x, y] in the source image
    cropped_images, crop_origins = pupil.crop_image_using_bounding_boxes(
        image=image,
        bounding_boxes=bounding_boxes,
    )

    # ===================== Log ================================================
    logger.success(f"Cropped {image} using bounding boxes")
    logger.success(f"Result: {cropped_images} into {len(cropped_images)} regions")
    logger.info(f"Crop origins [x, y] in the source image: {crop_origins.to_numpy().tolist()}")

    # ===================== Visualization  (Optional) ======================
    rr.init("crop_image_using_bounding_boxes_example", spawn=True)
    datatypes.visualize(image, entity_path="1-original")
    for i, cropped_image in enumerate(cropped_images):
        datatypes.visualize(cropped_image, entity_path=f"{i + 2}-Crop {i + 1}")

if __name__ == "__main__":
    crop_image_using_bounding_boxes_example()
