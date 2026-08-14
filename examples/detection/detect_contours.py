"""
Detect contours using a contour-based detector.
"""

from loguru import logger
import rerun as rr

from telekinesis import retina, datatypes


def detect_contours_example():
    """
    Detect contours using a contour-based detector.

    Extracts contours from the input image and returns contour using datatype `Contours`.
    """
    # ===================== Load Image ==========================================
    image_url = "https://assets.telekinesis.ai/examples/v1/images/nuts_scattered_filtered_gaussian.png"
    image = datatypes.Image.from_url(url=image_url).to_grayscale()

    # ===================== Run Skill ==========================================
    contours = retina.detect_contours(
        image=image,
        retrieval_mode="retrieve_list",
        approx_method="chain_approximate_simple",
        min_area=200,
        max_area=100000,
    )

    # ===================== Log ================================================
    logger.success(f"Detected contours in {image} using contour-based detector.")
    logger.success(f"Results: {contours}")

    logger.info(f"All detected contour points shape: {len(contours.points)}")
    logger.info(f"First detected contour: {contours[0]}")
    logger.info(f"First detected contour points shape: {contours[0].points.shape}")

    # ===================== Visualization  (Optional) ======================
    rr.init("detect_contours_example", spawn=True)
    datatypes.visualize(image, entity_path="/image")
    datatypes.visualize(contours, entity_path="/image/overlayed_contours")


if __name__ == "__main__":
    detect_contours_example()
