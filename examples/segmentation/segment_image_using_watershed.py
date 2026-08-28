"""Demonstrates watershed segmentation."""

import cv2
import numpy as np
from loguru import logger
import rerun as rr
import rerun.blueprint as rrb

from telekinesis import cornea, datatypes, pupil


def segment_image_using_watershed_example() -> None:
    """Segment an image using marker-controlled watershed."""

    # ===================== Load Image ==========================================
    image_url = (
        "https://assets.telekinesis.ai/examples/v1/images/water_coins.jpg"
    )
    image = datatypes.Image.from_url(url=image_url)
    image_np = image.to_numpy()
    gray_image = image.to_grayscale()

    # ===================== Build Markers =======================================
    markers_np = _build_watershed_markers(image_np)
    markers = datatypes.SegmentationImage(
        markers_np.astype(np.int32),
    )

    marker_labels, marker_counts = np.unique(
        markers_np,
        return_counts=True,
    )

    logger.info(f"Marker shape: {markers_np.shape}")
    logger.info(f"Marker dtype: {markers_np.dtype}")
    logger.info(f"Marker labels: {marker_labels}")
    logger.info(f"Marker counts: {marker_counts}")

    # ===================== Build Gradient Image ===============================
    gradient_y = pupil.filter_image_using_sobel(
        image=gray_image,
        dx=0,
        dy=1,
    ).to_numpy().astype(np.float32)

    gradient_x = pupil.filter_image_using_sobel(
        image=gray_image,
        dx=1,
        dy=0,
    ).to_numpy().astype(np.float32)

    # np.hypot computes sqrt(x**2 + y**2) without integer overflow.
    gradient = np.hypot(gradient_x, gradient_y)

    gradient_min = float(gradient.min())
    gradient_max = float(gradient.max())

    gradient_normalized = (
        (gradient - gradient_min)
        / (gradient_max - gradient_min + 1e-12)
        * 255.0
    ).astype(np.uint8)

    gradient_image = datatypes.Image(gradient_normalized)

    # ===================== Run Skill ==========================================
    segmented_image = cornea.segment_image_using_watershed(
        image=gradient_image,
        markers=markers,
        connectivity=1,
    )

    # ===================== Log ================================================
    logger.success(f"Segmented {image} using the watershed algorithm.")
    logger.success(f"Results: {segmented_image}")
    logger.info(
        f"Segmented image label codes: {segmented_image.label_codes}"
    )
    logger.info(
        "Segmented image number of labels: "
        f"{segmented_image.number_of_labels}"
    )
    logger.info(f"Segmented image shape: {segmented_image.shape}")
    logger.info(f"Segmented image dtype: {segmented_image.dtype}")

    # ===================== Visualization ======================================
    rr.init("segment_image_using_watershed_example", spawn=True)

    blueprint = rrb.Horizontal(
        rrb.Spatial2DView(
            origin="/input_image",
            name="Input",
        ),
        rrb.Spatial2DView(
            origin="/segmented_image",
            name="Watershed labels",
        ),
    )

    rr.send_blueprint(blueprint)

    datatypes.visualize(
        image,
        entity_path="/input_image",
    )
    datatypes.visualize(
        segmented_image,
        entity_path="/segmented_image",
    )


def _build_watershed_markers(
    rgb_image: np.ndarray,
    *,
    kernel_size: int = 3,
    opening_iterations: int = 2,
    dilate_iterations: int = 3,
    dist_fg_ratio: float = 0.7,
) -> np.ndarray:
    """Build marker IDs for marker-controlled watershed.

    Args:
        rgb_image: RGB image with shape `(H, W, 3)`, or a grayscale image
            with shape `(H, W)`.
        kernel_size: Width and height of the morphology kernel.
        opening_iterations: Number of morphological-opening iterations.
        dilate_iterations: Number of background-dilation iterations.
        dist_fg_ratio: Fraction of the maximum distance-transform value used
            to identify definite foreground pixels.

    Returns:
        An `int32` marker image. Label `0` represents unknown pixels, label
        `1` represents the known background, and labels `2...N` represent
        foreground components.
    """
    if rgb_image.ndim == 2:
        gray = rgb_image
    elif rgb_image.ndim == 3 and rgb_image.shape[2] == 3:
        gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    else:
        raise ValueError(
            "Expected an RGB image with shape (H, W, 3) or a grayscale "
            f"image with shape (H, W); got {rgb_image.shape}."
        )

    # Separate the dark coins from the bright background.
    _, thresholded = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )

    kernel = np.ones(
        (kernel_size, kernel_size),
        dtype=np.uint8,
    )

    # Remove small foreground noise.
    opening = cv2.morphologyEx(
        thresholded,
        cv2.MORPH_OPEN,
        kernel,
        iterations=opening_iterations,
    )

    # Region known to contain background.
    sure_background = cv2.dilate(
        opening,
        kernel,
        iterations=dilate_iterations,
    )

    # Separate touching objects by finding their central regions.
    distance = cv2.distanceTransform(
        opening,
        cv2.DIST_L2,
        5,
    )

    _, sure_foreground = cv2.threshold(
        distance,
        dist_fg_ratio * float(distance.max()),
        255,
        cv2.THRESH_BINARY,
    )

    sure_foreground = sure_foreground.astype(np.uint8)

    # Pixels that watershed must determine.
    unknown = cv2.subtract(
        sure_background,
        sure_foreground,
    )

    # Assign a unique marker ID to every foreground component.
    _, markers = cv2.connectedComponents(sure_foreground)

    # Reserve 0 for unknown pixels:
    # 0 = unknown, 1 = background, 2...N = foreground components.
    markers = markers.astype(np.int32) + 1
    markers[unknown == 255] = 0

    return markers


if __name__ == "__main__":
    segment_image_using_watershed_example()