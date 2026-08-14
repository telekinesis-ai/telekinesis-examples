"""
Detect circles using the classic Hough Circle Transform.
"""

from loguru import logger
import rerun as rr

from telekinesis import retina, datatypes


def detect_circle_using_classic_hough_example():
    """
    Detect circles using the classic Hough Circle Transform.

    Runs Hough circle detection on a grayscale image and returns circles using datatype `Circles`.
    """
    # ===================== Load Image ==========================================
    image_url = "https://assets.telekinesis.ai/examples/v1/images/metal_gears.jpg"
    image = datatypes.Image.from_url(url=image_url).to_grayscale()

    # ===================== Run Skill ==========================================
    circles = retina.detect_circle_using_classic_hough(
        image=image,
        inverse_resolution_ratio=1,
        min_distance=50,
        min_radius=40,
        max_radius=60,
        canny_detector_upper_threshold=300,
        accumulator_threshold=30,
    )

    # ===================== Log ================================================
    logger.success(f"Detected circles in {image} using classic Hough transform.")
    logger.success(f"Result: {circles}")

    logger.info(f"All detected circle centers shape: {circles.centers.shape}")
    logger.info(f"All detected circle radii shape: {circles.radii.shape}")

    # Access the first detected circle and log its details
    logger.info(f"First detected circle: {circles[0]}")
    logger.info(
        f"First detected circle center: {circles[0].center}, radius: {circles[0].radius}"
    )

    # ===================== Visualization  (Optional) ===========================
    rr.init("classic_hough_circle_detector_example", spawn=True)
    datatypes.visualize(image, entity_path="/image")
    datatypes.visualize(circles, entity_path="/image/detected-circles", label=[f"Circle {i}" for i in range(len(circles))])

if __name__ == "__main__":
    detect_circle_using_classic_hough_example()
