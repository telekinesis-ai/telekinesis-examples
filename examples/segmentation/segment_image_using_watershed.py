"""
Demonstrates watershed segmentation.
"""

import cv2
import numpy as np
from loguru import logger
import rerun as rr

from telekinesis import cornea, datatypes, pupil

def segment_image_using_watershed_example():
    """Segments an image using the watershed algorithm."""
    # ===================== Load Image ==========================================
    image_url = "https://assets.telekinesis.ai/examples/v1/images/water_coins.jpg"
    image = datatypes.Image.from_url(url=image_url)
    image_np = image.to_numpy()

    # ===================== Run Skill ==========================================
    markers = _build_watershed_markers(image_np.copy())

    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    gray_image = datatypes.Image(gray)
    gradient_y = pupil.filter_image_using_sobel(image=gray_image, dx=0, dy=1).to_numpy()
    gradient_x = pupil.filter_image_using_sobel(image=gray_image, dx=1, dy=0).to_numpy()
    gradient = np.sqrt(gradient_x**2 + gradient_y**2)
    gradient_normalized = (
        (gradient - gradient.min()) / (gradient.max() - gradient.min() + 1e-12) * 255
    ).astype(np.uint8)
    gradient_image = datatypes.Image(gradient_normalized)

    segmented_image = cornea.segment_image_using_watershed(
        image=gradient_image, markers=markers, connectivity=1
    )

    # ===================== Log ================================================
    logger.success(f"Segmented {image} using the watershed algorithm.")
    logger.success(f"Results: {segmented_image}")
    logger.info(f"Segmented image label codes: {segmented_image.label_codes}")
    logger.info(f"Segmented image number of labels: {segmented_image.number_of_labels}")
    logger.info(f"Segmented image shape: {segmented_image.shape}")
    logger.info(f"Segmented image dtype: {segmented_image.dtype}")

    # ===================== Visualization  (Optional) ======================
    rr.init("segment_image_using_watershed_example", spawn=True)
    datatypes.visualize(image, entity_path="/input_image")
    datatypes.visualize(segmented_image, entity_path="/segmented_image")


def _build_watershed_markers(rgb_image_np, kernel_size=3, opening_iterations=2,
                             dilate_iterations=3, dist_fg_ratio=0.7):
    """Builds watershed markers from an RGB image using morphological operations."""
    if rgb_image_np.ndim == 2:
        gray = rgb_image_np
    else:
        gray = cv2.cvtColor(rgb_image_np, cv2.COLOR_RGB2GRAY)

    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=opening_iterations)
    sure_bg = cv2.dilate(opening, kernel, iterations=dilate_iterations)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, dist_fg_ratio * dist_transform.max(), 255, 0)
    sure_fg_u8 = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg_u8)
    num_labels, markers = cv2.connectedComponents(sure_fg_u8)
    markers = markers + 1
    markers[unknown == 255] = 0
    return markers.astype(np.int32)


if __name__ == "__main__":
    segment_image_using_watershed_example()
