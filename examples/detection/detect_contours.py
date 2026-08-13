"""
Detect contours using a contour-based detector.

Extracts contours from the input image and returns
coco-style annotations.

The annotations are used for visualization overlays.
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
    logger.info(f"Loaded {image} from the URL: {image_url}")

    # ===================== Run Skill ==========================================
    contours = retina.detect_contours(
        image=image,
        retrieval_mode="retrieve_list",
        approx_method="chain_approximate_simple",
        min_area=200,
        max_area=100000,
    )
    logger.info(f"Detected Contours: {contours}")

    # Access the underlying grouped data
    all_contour_points = contours.data
    logger.info(f"All detected contour points: {all_contour_points}")

    # Access the first detected contour and log its details
    if contours:
        first_contour = contours[0]
        logger.info(f"First detected contour: {first_contour}")
        first_contour_points_list = first_contour.data
        logger.info(f"First detected contour points shape: {first_contour_points_list.shape}")

    # ===================== Visualization  (Optional) ======================
    rr.init("detect_contours_example", spawn=True)
    # No label for contours, check.
    datatypes.visualize(
        image,
        contours,
        entity_path="/Image/overlayed_contours",
        label=[f"Contour {i}" for i in range(len(contours))],
    )



if __name__ == "__main__":
    detect_contours_example()
