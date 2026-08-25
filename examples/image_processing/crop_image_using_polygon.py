"""Demonstrates cropping an image using a polygon mask."""

from loguru import logger
import rerun as rr

from telekinesis import pupil, datatypes


def crop_image_using_polygon_example():
    """Crops image using a polygon mask."""
    # ===================== Load Image ==========================================
    image_url = "https://assets.telekinesis.ai/examples/v1/images/pedestrians.jpg"
    image = datatypes.Image.from_url(image_url)

    # ===================== Run Skill ==========================================
    # Define polygon vertices in the format [[x1, y1], [x2, y2], ..., [xn, yn]]
    polygon_vertices = [
        [37, 404],
        [46, 373],
        [74, 323],
        [106, 258],
        [125, 154],
        [165, 106],
        [200, 115],
        [210, 173],
        [206, 199],
        [250, 208],
        [193, 255],
        [216, 331],
        [240, 383],
        [250, 411],
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
