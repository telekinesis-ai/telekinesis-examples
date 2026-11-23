"""Demonstrates cropping an image using a polygon mask."""

from loguru import logger
import rerun as rr

from telekinesis import pupil, datatypes


def crop_image_using_polygon_example():
    """Crops image using a polygon mask."""
    # ===================== Load Image ==========================================
    # Example data: XYZ-IBD (CC BY-NC-SA 4.0), modified. Credits and licences: THIRD_PARTY_DATA.md
    image_url = "https://assets.telekinesis.ai/examples/v1/images/xyzibd_brackets_yellow_bin_rgb.jpg"
    image = datatypes.Image.from_url(image_url)

    # ===================== Run Skill ==========================================
    # Define polygon vertices in the format [[x1, y1], [x2, y2], ..., [xn, yn]]
    polygon_vertices = [
        [250, 95],
        [272, 72],
        [775, 70],
        [797, 92],
        [797, 410],
        [775, 432],
        [272, 432],
        [250, 410],
    ]

    filtered_image = pupil.crop_image_using_polygon(
        image=image,
        polygon_vertices=polygon_vertices,
    )

    # ===================== Log ================================================
    logger.success(f"Cropped {image} using polygon")
    logger.success(f"Result: {filtered_image}")

    # ===================== Visualization  (Optional) ======================
    rr.init("crop_image_using_polygon_example", spawn=True)
    datatypes.visualize(image, entity_path="1-original")
    datatypes.visualize(filtered_image, entity_path="2-cropped")

if __name__ == "__main__":
    crop_image_using_polygon_example()
