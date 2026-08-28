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
    gray_image = image.to_grayscale()

    # ===================== Build Markers =======================================
    markers = _build_watershed_markers(
        gray_image=gray_image,
        kernel_size=3,
        opening_iterations=2,
        dilate_iterations=3,
        dist_fg_ratio=0.7,
    )

    # ===================== Build Gradient =====================================
    gradient_y = pupil.filter_image_using_sobel(
        image=gray_image,
        dx=0,
        dy=1,
        output_format="32bit",
    ).to_numpy()

    gradient_x = pupil.filter_image_using_sobel(
        image=gray_image,
        dx=1,
        dy=0,
        output_format="32bit",
    ).to_numpy()

    gradient = np.hypot(
        gradient_x.astype(np.float32),
        gradient_y.astype(np.float32),
    )

    gradient_image = datatypes.Image(gradient)

    gradient_normalized = pupil.normalize_image_intensity(
        image=gradient_image,
        alpha=0.0,
        beta=255.0,
        normalization_method="minmax",
        output_format="8bit",
    )

    # ===================== Run Watershed ======================================
    segmented_image = cornea.segment_image_using_watershed(
        image=gradient_normalized,
        markers=markers,
        connectivity=1,
    )

    # ===================== Log ================================================
    logger.success(f"Segmented {image} using watershed.")
    logger.success(f"Result: {segmented_image}")
    logger.info(f"Labels: {segmented_image.label_codes}")
    logger.info(f"Number of labels: {segmented_image.number_of_labels}")

    # ===================== Visualization ======================================
    rr.init("segment_image_using_watershed_example", spawn=True)

    rr.send_blueprint(
        rrb.Horizontal(
            rrb.Spatial2DView(origin="/input", name="Input"),
            rrb.Spatial2DView(origin="/labels", name="Watershed labels"),
        )
    )

    datatypes.visualize(image, entity_path="/input")
    datatypes.visualize(segmented_image, entity_path="/labels")


def _build_watershed_markers(
    *,
    gray_image: datatypes.Image,
    kernel_size: int,
    opening_iterations: int,
    dilate_iterations: int,
    dist_fg_ratio: float,
) -> datatypes.SegmentationImage:
    """Build marker IDs for watershed segmentation."""

    # Cornea Otsu returns 0=background, 1=foreground.
    otsu_mask = cornea.segment_image_using_otsu_threshold(
        image=gray_image,
    )

    # Coins are dark, so invert the Otsu result and convert to 0/255.
    thresholded = datatypes.Image(
        ((1 - otsu_mask.to_numpy()) * 255).astype(np.uint8)
    )

    # Equivalent to cv2.MORPH_OPEN with a square kernel.
    opening = pupil.filter_image_using_morphological_open(
        image=thresholded,
        kernel_size=kernel_size,
        kernel_shape="rectangle",
        iterations=opening_iterations,
        border_type="constant",
        border_value=0,
    )

    sure_background = pupil.filter_image_using_morphological_dilate(
        image=opening,
        kernel_size=kernel_size,
        kernel_shape="rectangle",
        iterations=dilate_iterations,
        border_type="constant",
        border_value=0,
    )

    distance = cv2.distanceTransform(
        opening.to_numpy(),
        cv2.DIST_L2,
        5,
    )
    sure_foreground = (
        distance > dist_fg_ratio * float(distance.max())
    ).astype(np.uint8) * 255
    sure_foreground_image = datatypes.Image(sure_foreground)

    unknown = pupil.bitwise_difference_images(
        image_a=sure_background,
        image_b=sure_foreground_image,
    ).to_numpy()

    _, markers = cv2.connectedComponents(
        sure_foreground,
        connectivity=8,
    )
    markers = markers.astype(np.int32) + 1
    markers[unknown != 0] = 0

    return datatypes.SegmentationImage(markers)


if __name__ == "__main__":
    segment_image_using_watershed_example()