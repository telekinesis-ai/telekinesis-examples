"""
Demonstrates watershed segmentation.

This example:
- Downloads an example image.
- Creates watershed markers using morphological operations.
- Segments the image using the watershed algorithm.
- Visualizes the result using Rerun.
"""

import numpy as np
import requests
import cv2
from loguru import logger
import rerun as rr
import rerun.blueprint as rrb
from datatypes import datatypes
from telekinesis import cornea, pupil

def segment_image_using_watershed_example():
    """Segments an image using the watershed algorithm."""
    # ===================== Load Image ==========================================
    image_url = "https://telekinesis-public-assets.s3.us-east-1.amazonaws.com/examples/v1/images/water_coins.jpg"
    original_image = fetch_image(image_url)
    original_np = original_image.to_numpy()

    # ===================== Run Skill ==========================================
    markers = _build_watershed_markers(original_np.copy())
    logger.info(f"Markers computed with dtype={markers.dtype} min={markers.min()} max={markers.max()}")

    if original_np.ndim == 3:
        gray = cv2.cvtColor(original_np, cv2.COLOR_RGB2GRAY)
    else:
        gray = original_np

    gray_image = datatypes.Image(image=gray)
    gradient_y = pupil.filter_image_using_sobel(gray_image, dx=0, dy=1).to_numpy()
    gradient_x = pupil.filter_image_using_sobel(gray_image, dx=1, dy=0).to_numpy()
    gradient = np.sqrt(gradient_x**2 + gradient_y**2)
    gradient_normalized = ((gradient - gradient.min()) / (gradient.max() - gradient.min() + 1e-12) * 255).astype(np.uint8)
    gradient_image = datatypes.Image(image=gradient_normalized, color_model="L")

    annotations = cornea.segment_image_using_watershed(image=gradient_image, markers=markers, connectivity=1)
    annotations_dict = annotations.to_dict()
    logger.success("Segmentation completed.")

    # ===================== Visualization (Optional) ======================
    visualize(original_np, annotations_dict)


def fetch_image(image_url: str) -> datatypes.Image:
    """
    Downloads an image from a given URL and returns it as a telekinesis.datatypes.Image object.
    """
    response = requests.get(image_url, timeout=60)
    response.raise_for_status()
    image_bgr = cv2.imdecode(
        np.frombuffer(response.content, dtype=np.uint8), cv2.IMREAD_COLOR,
    )
    original_image = datatypes.Image(image=image_bgr, color_model="BGR")
    original_image = pupil.convert_image_color_space(
        original_image, source_color_space="BGR", target_color_space="RGB"
    )
    logger.success(f"Loaded image from {image_url}")
    return original_image

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

def visualize(original_np: np.ndarray, annotations_dict: dict) -> None:
    """Visualizes the original image and the segmentation result using Rerun."""
    rr.init("cornea_watershed_segmentation", spawn=True)
    rr.send_blueprint(
        rrb.Blueprint(
            rrb.Grid(
                rrb.Spatial2DView(name="Original", origin="original"),
                rrb.Spatial2DView(name="Overlayed Image", origin="overlayed"),
            ),
            rrb.SelectionPanel(),
            rrb.TimePanel(),
        ),
        make_active=True,
    )
    mask_np = annotations_dict["labeled_mask"]
    img = original_np.copy()
    img[mask_np == 0] = [255, 0, 0]
    rr.log("original", rr.Image(original_np))
    rr.log("overlayed", rr.Image(img))

if __name__ == "__main__":
    segment_image_using_watershed_example()
