"""
    This example demonstrates how to use the Cornea SDK for image segmentation operations, 
    including color-based segmentation, region-based segmentation, thresholding, and superpixel segmentation.
    
    It also shows how to visualize results using Rerun.
"""

import argparse
import difflib
import pathlib
from typing import Optional
import numpy as np

import cv2
from loguru import logger
import rerun as rr
import rerun.blueprint as rrb
from pycocotools import mask as mask_utils

from datatypes import datatypes, io
from telekinesis import cornea, pupil

ROOT_DIR = pathlib.Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "telekinesis-data"



# ===================== Color Examples =====================


def segment_image_using_rgb_example():
    """
    Performs RGB color space segmentation.

    This function demonstrates how to segment an image using a specified
    RGB range.

    The returned annotations is processed and used for visualization.
    """
    # ===================== Operation ==========================================

    # Load image
    filepath = str(DATA_DIR / "images" / "cylinder_on_conveyor.jpg")
    image = io.load_image(filepath=filepath)
    logger.success(f"Loaded image from {filepath}")

    # Segment image
    result = cornea.segment_image_using_rgb(
        image=image,
        lower_bound=(0, 50, 50),
        upper_bound=(180, 255, 255)
    )

    # Access results
    annotation = result.to_dict()
    logger.success("Segmentation completed.")

    # ===================== Visualization  (Optional) ======================

    # Initialize Rerun for visualization
    rr.init("cornea_rgb_segmentation", spawn=True)
    rr.send_blueprint(
        rrb.Blueprint(
            rrb.Grid(
                rrb.Spatial2DView(name="Original", origin="input"),
                rrb.Spatial2DView(name="Mask", origin="segmented_mask"),
            ),
            rrb.SelectionPanel(),
            rrb.TimePanel(),
        ),
        make_active=True,
    )

    # Input image
    in_np_image = image.to_numpy()

    # Segmentation mask
    mask_np = annotation['labeled_mask']

    # Log images
    rr.log("input", rr.Image(in_np_image))
    rr.log("segmented_mask", rr.Image(mask_np))


def segment_image_using_hsv_example():
    """
    Performs HSV color space segmentation.

    This function demonstrates how to segment an image using a specified
    HSV range.
    
    The returned annotations is processed and used for visualization.
    """

    # ===================== Operation ==========================================

    # Load image
    filepath = str(DATA_DIR / "images" / "wires_rgb.png")
    image = io.load_image(filepath=filepath)
    logger.success(f"Loaded image from {filepath}")


    # Segment image
    result = cornea.segment_image_using_hsv(
        image=image,
        lower_bound=(0, 50, 50),
        upper_bound=(180, 255, 255)
    )

    # Access results
    annotation = result.to_dict()
    logger.success("Segmentation completed.")

    # ===================== Visualization  (Optional) ======================

    # Initialize Rerun for visualization
    rr.init("cornea_hsv_segmentation", spawn=True)
    rr.send_blueprint(
        rrb.Blueprint(
            rrb.Grid(
                rrb.Spatial2DView(name="Original", origin="input"),
                rrb.Spatial2DView(name="Mask", origin="segmented_mask"),
            ),
            rrb.SelectionPanel(),
            rrb.TimePanel(),
        ),
        make_active=True,
    )

    # Input image
    in_np_image = image.to_numpy()

    # Segmentation mask
    mask_np = annotation['labeled_mask']

    # Log images
    rr.log("input", rr.Image(in_np_image))
    rr.log("segmented_mask", rr.Image(mask_np))


def segment_image_using_lab_example():
    """
    Performs LAB color space segmentation.

    This function demonstrates how to segment an image using a specified
    LAB range, and visualize it both on disk and using the Rerun logging framework.

    The returned annotations is processed and used for visualization.
    """
    # ===================== Operation ==========================================

    # Load image
    filepath = str(DATA_DIR / "images" / "car_painting.jpg")
    image = io.load_image(filepath=filepath)
    logger.success(f"Loaded image from {filepath}")

    # Segment image
    result = cornea.segment_image_using_lab(
        image=image,
        lower_bound=(120, 50, 50),
        upper_bound=(180, 255, 255)
    )

    # Access results
    annotation = result.to_dict()
    logger.success("Segmentation completed.")

    # ===================== Visualization  (Optional) ======================

    # Initialize Rerun for visualization
    rr.init("cornea_lab_segmentation", spawn=True)
    rr.send_blueprint(
        rrb.Blueprint(
            rrb.Grid(
                rrb.Spatial2DView(name="Original", origin="input"),
                rrb.Spatial2DView(name="Mask", origin="segmented_mask"),
            ),
            rrb.SelectionPanel(),
            rrb.TimePanel(),
        ),
        make_active=True,
    )

    # Input image
    in_np_image = image.to_numpy()

    # Segmentation mask
    mask_np = annotation['labeled_mask']

    # Log images
    rr.log("input", rr.Image(in_np_image))
    rr.log("segmented_mask", rr.Image(mask_np))


def segment_image_using_ycrcb_example():
    """
    Performs YCrCb color space segmentation.

    This function demonstrates how to segment an image using a specified
    YCrCb range.

    The returned annotations is processed and used for visualization.
    """
    # ===================== Operation ==========================================

    # Load image
    filepath = str(DATA_DIR / "images" / "David_Schwimmer.jpg")
    image = io.load_image(filepath=filepath)
    logger.success(f"Loaded image from {filepath}")

    # Segment image
    result = cornea.segment_image_using_ycrcb(
        image=image,
        lower_bound=(0, 133, 77),
        upper_bound=(255, 173, 127)
    )

    # Access results
    annotation = result.to_dict()
    logger.success("Segmentation completed.")

    # ===================== Visualization  (Optional) ======================

    # Initialize Rerun for visualization
    rr.init("cornea_ycrcb_segmentation", spawn=True)
    rr.send_blueprint(
        rrb.Blueprint(
            rrb.Grid(
                rrb.Spatial2DView(name="Original", origin="input"),
                rrb.Spatial2DView(name="Mask", origin="segmented_mask"),
            ),
            rrb.SelectionPanel(),
            rrb.TimePanel(),
        ),
        make_active=True,
    )

    # Input image
    in_np_image = image.to_numpy()

    # Segmentation mask
    mask_np = annotation['labeled_mask']

    # Log images
    rr.log("input", rr.Image(in_np_image))
    rr.log("segmented_mask", rr.Image(mask_np))


# ===================== Region Examples =====================

def segment_image_using_focus_region_example():
    """
    Segments the in-focus regions of an image.

    This function demonstrates how to segment the in-focus regions of an image
    using a specified blur kernel size and threshold.

    The returned annotations is processed and used for visualization.
    """

    # ===================== Operation ==========================================

    # Load image
    filepath = str(DATA_DIR / "images" / "matt_leblanc.jpg")
    image = io.load_image(filepath=filepath)
    logger.success(f"Loaded image from {filepath}")

    # Segment image
    result = cornea.segment_image_using_focus_region(
        image=image,
        blur_kernel_size=151,
        threshold=5
    )

    # Access results
    annotation = result.to_dict()

    # ===================== Visualization  (Optional) ======================

    # Initialize Rerun for visualization
    rr.init("cornea_focus_region_segmentation", spawn=True)
    rr.send_blueprint(
        rrb.Blueprint(
            rrb.Grid(
                rrb.Spatial2DView(name="Original", origin="input"),
                rrb.Spatial2DView(name="Mask", origin="segmented_mask"),
            ),
            rrb.SelectionPanel(),
            rrb.TimePanel(),
        ),
        make_active=True,
    )

    # Input image
    in_np_image = image.to_numpy()

    # Segmentation mask
    mask_np = annotation['labeled_mask']

    # Log images
    rr.log("input", rr.Image(in_np_image))
    rr.log("segmented_mask", rr.Image(mask_np))


def segment_image_using_watershed_example():
    """
    Performs segmentation using watershed.

    This function demonstrates how to perform watershed segmentation on an image
    using markers, and also one of the ways how to create the markers.

    The returned annotations is processed and used for visualization.
    """
    # ===================== Operation ==========================================

    # Load original image
    filepath = str(DATA_DIR / "images" / "water_coins.jpg")
    original_image = io.load_image(filepath=filepath, as_binary=False)
    original_np = original_image.to_numpy()
    logger.success(f"Loaded image from {filepath}")

    # Helper function to build watershed markers
    def _build_and_save_watershed_markers_opencv(
        *,
        rgb_image_np: np.ndarray,
        kernel_size: int = 3,
        opening_iterations: int = 2,
        dilate_iterations: int = 3,
        dist_fg_ratio: float = 0.7,
        save_debug: bool = False,
    ):
        """
        Build markers using the OpenCV tutorial pipeline and return them.

        Marker pipeline:
        1) gray
        2) Otsu threshold (binary inverse)
        3) opening
        4) sure_bg via dilation
        5) sure_fg via distanceTransform + threshold (ratio of max)
        6) unknown = sure_bg - sure_fg
        7) connectedComponents(sure_fg) -> markers
            markers += 1
            markers[unknown==255] = 0

        Returns:
        - markers as int32 array (ready for watershed / your wrapper)
        - optionally debug intermediates as a dictionary
        """
        if rgb_image_np.ndim == 2:
            bgr = cv2.cvtColor(rgb_image_np, cv2.COLOR_GRAY2BGR)
            gray = rgb_image_np
        else:
            bgr = cv2.cvtColor(rgb_image_np, cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        # 1) Otsu threshold (inverse)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # 2) Opening
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=opening_iterations)

        # 3) Sure background
        sure_bg = cv2.dilate(opening, kernel, iterations=dilate_iterations)

        # 4) Sure foreground
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist_transform, dist_fg_ratio * dist_transform.max(), 255, 0)
        sure_fg_u8 = np.uint8(sure_fg)

        # 5) Unknown
        unknown = cv2.subtract(sure_bg, sure_fg_u8)

        # 6) connectedComponents -> markers
        num_labels, markers = cv2.connectedComponents(sure_fg_u8)
        markers = markers + 1
        markers[unknown == 255] = 0

        markers_i32 = markers.astype(np.int32)

        debug_data = None
        if save_debug:
            debug_data = dict(
                gray=gray,
                thresh=thresh,
                opening=opening,
                sure_bg=sure_bg,
                dist_transform=dist_transform.astype(np.float32),
                sure_fg=sure_fg_u8,
                unknown=unknown,
                markers=markers_i32,
                num_labels=np.array([num_labels], dtype=np.int32),
            )
        return markers_i32, debug_data

    # Build watershed markers 
    markers, debug_data = _build_and_save_watershed_markers_opencv(
        rgb_image_np=original_np.copy(),
        save_debug=True,
    )
    logger.info("Markers computed with dtype=%s min=%d max=%d", markers.dtype, int(markers.min()), int(markers.max()))

    # --- Build elevation/gradient image
    if original_np.ndim == 3:
        gray = cv2.cvtColor(original_np, cv2.COLOR_RGB2GRAY)
    else:
        gray = original_np

    # Using telekinesis pupil library to get the gradient image.
    gray_image = datatypes.Image(image=gray)
    gradient_y = pupil.filter_image_using_sobel(gray_image, dx=0, dy=1).to_numpy()
    gradient_x = pupil.filter_image_using_sobel(gray_image, dx=1, dy=0).to_numpy()

    gradient = np.sqrt(gradient_x**2 + gradient_y**2)

    gradient_normalized = (
        (gradient - gradient.min()) / (gradient.max() - gradient.min() + 1e-12) * 255
    ).astype(np.uint8)
    gradient_image = datatypes.Image(image=gradient_normalized, color_model="L")

    result = cornea.segment_image_using_watershed(
        image=gradient_image,   
        markers=markers,
        connectivity=1
    )

    annotation = result.to_dict()
    logger.success("Segmentation completed.")

    # ===================== Visualization (Optional) ======================

    # Initialize Rerun for visualization
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

    # Segmentation mask
    mask_np = annotation["labeled_mask"]

    img = original_np.copy()
    img[mask_np == 0] = [255, 0, 0]

    # Log images
    rr.log("original", rr.Image(original_np))
    rr.log("overlayed", rr.Image(img))


def segment_image_using_flood_fill_example():
    """
    Performs flood fill segmentation.

    This function demonstrates how to perform flood fill segmentation on an image
    using a specified seed point and tolerance.

    The returned annotations is processed and used for visualization.
    """
    # ===================== Operation ==========================================

    # Load image
    filepath = str(DATA_DIR / "images" / "erode.jpg")
    image = io.load_image(filepath=filepath)
    logger.success(f"Loaded image from {filepath}")

    # Perform flood fill segmentation
    result = cornea.segment_image_using_flood_fill(
        image=image,
        seed_point=(0, 0),
        tolerance=10
    )

    # Access results
    annotation = result.to_dict()
    logger.success("Segmentation completed.")

    # ===================== Visualization  (Optional) ======================

    # Initialize Rerun for visualization
    rr.init("cornea_flood_fill_segmentation", spawn=True)
    rr.send_blueprint(
        rrb.Blueprint(
            rrb.Grid(
                rrb.Spatial2DView(name="Original", origin="input"),
                rrb.Spatial2DView(name="Mask", origin="segmented_mask"),
            ),
            rrb.SelectionPanel(),
            rrb.TimePanel(),
        ),
        make_active=True,
    )

    # Input image
    in_np_image = image.to_numpy()

    # Segmentation mask
    mask_np = annotation['labeled_mask']

    # Log images
    rr.log("input", rr.Image(in_np_image))
    rr.log("segmented_mask", rr.Image(mask_np))


# ===================== Threshold Examples =====================

def segment_image_using_otsu_threshold_example():
    """
    Performs Otsu threshold segmentation.

    This function applies Otsu's method to find a global threshold for the image.

    The returned annotations is processed and used for visualization.
    """
    # ===================== Operation ==========================================

    # Load image
    filepath = str(DATA_DIR / "images" / "buttons_arranged.jpg")
    image = io.load_image(filepath=filepath, as_gray=True)
    logger.success(f"Loaded image from {filepath}")

    # Apply Otsu's thresholding
    annotation = cornea.segment_image_using_otsu_threshold(image=image)

    # Access results
    annotation = annotation.to_dict()
    logger.success("Segmentation completed.")

    # ===================== Visualization  (Optional) ======================

    # Initialize Rerun for visualization
    rr.init("cornea_otsu_threshold_segmentation", spawn=True)
    rr.send_blueprint(
        rrb.Blueprint(
            rrb.Grid(
                rrb.Spatial2DView(name="Input", origin="input"),
                rrb.Spatial2DView(name="Mask", origin="segmented_mask"),
            ),
            rrb.SelectionPanel(),
            rrb.TimePanel(),
        ),
        make_active=True,
    )

    # Input image
    in_np_image = image.to_numpy()

    # Mask
    mask_np = annotation['labeled_mask']

    # Log images
    rr.log("input", rr.Image(in_np_image))
    rr.log("segmented_mask", rr.SegmentationImage(mask_np))


def segment_image_using_local_threshold_example():
    """
    Performs local threshold segmentation.

    This function applies adaptive (local) thresholding to segment images where
    illumination varies across the scene.

    The returned annotations is processed and used for visualization.
    """
    # ===================== Operation ==========================================

    # Load image
    filepath = str(DATA_DIR / "images" / "car_number_plate.jpg")
    image = io.load_image(filepath=filepath, as_gray=True)
    logger.success(f"Loaded image from {filepath}")

    # Segment image
    result = cornea.segment_image_using_local_threshold(
        image=image,
        block_size=23
    )

    # Access results
    annotation = result.to_dict()
    logger.success("Segmentation completed.")

    # ===================== Visualization  (Optional) ======================

    # Initialize Rerun for visualization
    rr.init("cornea_local_threshold_segmentation", spawn=True)
    rr.send_blueprint(
        rrb.Blueprint(
            rrb.Grid(
                rrb.Spatial2DView(name="Original", origin="input"),
                rrb.Spatial2DView(name="Mask", origin="segmented_mask"),
            ),
            rrb.SelectionPanel(),
            rrb.TimePanel(),
        ),
        make_active=True,
    )


    # Input image
    in_np_image = image.to_numpy()

    # Mask
    mask_np = annotation['labeled_mask']

    # Rerun logging
    rr.log("input", rr.Image(in_np_image))
    rr.log("segmented_mask", rr.Image(mask_np))


def segment_image_using_yen_threshold_example():
    """
    Performs Yen threshold segmentation.

    This function applies Yen's method to segment images based on their intensity histograms.

    The returned annotations is processed and used for visualization.
    """
    # ===================== Operation ==========================================

    # Load image
    filepath = str(DATA_DIR / "images" / "bolts_and_ nuts_white_bg.jpg")
    image = io.load_image(filepath=filepath, as_gray=True)
    logger.success(f"Loaded image from {filepath}")

    # Segment image
    result = cornea.segment_image_using_yen_threshold(
        image=image
    )

    # Access results
    annotation = result.to_dict()
    logger.success("Segmentation completed.")

    # ===================== Visualization  (Optional) ======================

    # Initialize Rerun for visualization
    rr.init("cornea_yen_threshold_segmentation", spawn=True)
    rr.send_blueprint(
        rrb.Blueprint(
            rrb.Grid(
                rrb.Spatial2DView(name="Original", origin="input"),
                rrb.Spatial2DView(name="Mask", origin="segmented_mask"),
            ),
            rrb.SelectionPanel(),
            rrb.TimePanel(),
        ),
        make_active=True,
    )


    # Input image
    in_np_image = image.to_numpy()

    # Mask
    mask_np = annotation['labeled_mask']

    # Rerun logging
    rr.log("input", rr.Image(in_np_image))
    rr.log("segmented_mask", rr.Image(mask_np))


def segment_image_using_threshold_example():
    """
    Performs basic threshold segmentation.

    This function applies a simple global threshold to segment images based on their intensity values.

    The returned annotations is processed and used for visualization.
    """
    # ===================== Operation ==========================================

    # Load image
    filepath = str(DATA_DIR / "images" / "nuts_scattered.jpg")
    image = io.load_image(filepath=filepath, as_gray=True)
    logger.success(f"Loaded image from {filepath}")

    # Apply thresholding
    result = cornea.segment_image_using_threshold(
        image=image,
        min_value=45,
        max_value=255,
        threshold_type='binary',
    )

    # Access results
    annotation = result.to_dict()
    logger.success("Segmentation completed.")

    # ===================== Visualization  (Optional) ======================

    # Initialize Rerun for visualization
    rr.init("cornea_threshold_segmentation", spawn=True)
    rr.send_blueprint(
        rrb.Blueprint(
            rrb.Grid(
                rrb.Spatial2DView(name="Original", origin="input"),
                rrb.Spatial2DView(name="Mask", origin="segmented_mask"),
            ),
            rrb.SelectionPanel(),
            rrb.TimePanel(),
        ),
        make_active=True,
    )

    # Input image
    in_np_image = image.to_numpy()

    # Mask
    mask_np = annotation['labeled_mask']

    # Rerun logging
    rr.log("input", rr.Image(in_np_image))
    rr.log("segmented_mask", rr.Image(mask_np))


def segment_image_using_adaptive_threshold_example():
    """
    Performs adaptive threshold segmentation.

    This function applies adaptive thresholding to segment images with 
    non-uniform illumination by computing local thresholds.

    The returned annotations is processed and used for visualization.
    """
    # ===================== Operation ==========================================

    # Load image
    filepath = str(DATA_DIR / "images" / "car_number_plate.jpg")
    image = io.load_image(filepath=filepath, as_gray=True)
    logger.success(f"Loaded image from {filepath}")

    # Apply adaptive thresholding
    result = cornea.segment_image_using_adaptive_threshold(
        image=image,
        max_value=255,
        adaptive_method="gaussian constant",
        threshold_type='binary',
        block_size=61,
        offset_constant=5
    )

    # Access results
    annotation = result.to_dict()
    logger.success("Segmentation completed.")

    # ===================== Visualization  (Optional) ======================

    # Initialize Rerun for visualization
    rr.init("cornea_adaptive_threshold_segmentation", spawn=True)
    rr.send_blueprint(
        rrb.Blueprint(
            rrb.Grid(
                rrb.Spatial2DView(name="Original", origin="input"),
                rrb.Spatial2DView(name="Mask", origin="segmented_mask"),
            ),
            rrb.SelectionPanel(),
            rrb.TimePanel(),
        ),
        make_active=True,
    )

    # Input image
    in_np_image = image.to_numpy()

    # Mask
    mask_np = annotation['labeled_mask']

    # Rerun logging
    rr.log("input", rr.Image(in_np_image))
    rr.log("segmented_mask", rr.Image(mask_np))


def segment_image_using_laplacian_threshold_example():
    """
    Performs Laplacian threshold segmentation. segment_image_using_laplacian_threshold

    This function uses the Laplacian operator to highlight regions with high
    second-order intensity changes and applies thresholding to segment sharp or
    edge-rich areas.

    The returned annotations is processed and used for visualization.
    """
    # ===================== Operation ==========================================

    # Load image
    filepath = str(DATA_DIR / "images" / "mechanical_parts_gray.png")
    image = io.load_image(filepath, as_gray=True)
    logger.success(f"Loaded image from {filepath}")

    # Apply Laplacian thresholding
    result = cornea.segment_image_using_laplacian_threshold(image=image)

    # Access results
    annotation = result.to_dict()
    logger.success("Segmentation completed.")

    # ===================== Visualization  (Optional) ======================

    # Initialize Rerun for visualization
    rr.init("cornea_laplacian_threshold_segmentation", spawn=True)
    rr.send_blueprint(
        rrb.Blueprint(
            rrb.Grid(
                rrb.Spatial2DView(name="Original", origin="input"),
                rrb.Spatial2DView(name="Mask", origin="segmented_mask"),
            ),
            rrb.SelectionPanel(),
            rrb.TimePanel(),
        ),
        make_active=True,
    )

    # Input image
    in_np_image = image.to_numpy()

    # Mask
    mask_np = annotation['labeled_mask']

    # Rerun logging
    rr.log("input", rr.Image(in_np_image))
    rr.log("segmented_mask", rr.Image(mask_np))


# ===================== Superpixel Examples =====================

def segment_image_using_felzenszwalb_example():
    """
    Performs Felzenszwalb segmentation.

    This function applies the Felzenszwalb algorithm to segment an image into
    regions based on color similarity and spatial proximity. It is useful for
    identifying distinct objects or regions within an image.

    The returned annotations is processed and used for visualization.
    """
    # ===================== Operation ==========================================

    # Load image
    filepath = str(DATA_DIR / "images" / "eggs_carton.jpg")
    image = io.load_image(filepath=filepath)
    logger.success(f"Loaded image from {filepath}")

    # Segment Image
    result = cornea.segment_image_using_felzenszwalb(
        image=image,
        scale=500,
        sigma=1,
        min_size=200
    )

    # Access results
    annotation = result.to_dict()
    logger.success("Segmentation completed.")

    # ===================== Visualization  (Optional) ======================

    # Initialize Rerun for visualization
    rr.init("cornea_felzenszwalb_segmentation", spawn=True)
    rr.send_blueprint(
        rrb.Blueprint(
            rrb.Grid(
                rrb.Spatial2DView(name="Original", origin="input"),
                rrb.Spatial2DView(name="Mask", origin="segmented_mask"),
            ),
            rrb.SelectionPanel(),
            rrb.TimePanel(),
        ),
        make_active=True,
    )

    # Input image
    in_np_image = image.to_numpy()

    # Segmentation Mask
    mask_np = annotation['labeled_mask']

    # Log Images
    rr.log("input", rr.Image(in_np_image))
    rr.log("segmented_mask", rr.SegmentationImage(mask_np))


def segment_image_using_slic_superpixel_example():
    """
    Performs SLIC superpixel segmentation.

    This function segments an image into compact, approximately uniform superpixels
    using the SLIC algorithm. Superpixels preserve object boundaries while reducing
    image complexity, making them useful for robotic perception, region grouping,
    and downstream segmentation or classification tasks.

    The returned annotations is processed and used for visualization.
    """
    # ===================== Operation ==========================================

    # Load image
    filepath = str(DATA_DIR / "images" / "nuts.jpg")
    image = io.load_image(filepath=filepath)
    logger.success(f"Loaded image from {filepath}")

    # Segment Image
    result = cornea.segment_image_using_slic_superpixel(
        image=image,
        num_segments=2,
        compactness=15.0,
        max_iterations=20,
        sigma=0.0,
        spacing=None,
        convert_to_lab=None,
        enforce_connectivity=True,
        min_size_factor=0.1,
        max_size_factor=150.0,
        use_slic_zero=False,
        start_label=1,
        mask=None,
        channel_axis=-1
    )

    # Access results
    annotation = result.to_dict()
    logger.success("Segmentation completed.")

    # ===================== Visualization  (Optional) ======================

    # Initialize Rerun for visualization
    rr.init("cornea_slic_superpixel_segmentation", spawn=True)
    rr.send_blueprint(
        rrb.Blueprint(
            rrb.Grid(
                rrb.Spatial2DView(name="Original", origin="input"),
                rrb.Spatial2DView(name="Mask", origin="segmented_mask"),
            ),
            rrb.SelectionPanel(),
            rrb.TimePanel(),
        ),
        make_active=True,
    )

    # Input Image
    in_np_image = image.to_numpy()

    # Segmentation Mask
    mask_np = annotation['labeled_mask']

    # Log Images
    rr.log("input", rr.Image(in_np_image))
    rr.log("segmented_mask", rr.SegmentationImage(mask_np))


def filter_segments_by_area_example():
    """
    Filters superpixels based on area.

    This function filters out superpixels that do not meet the specified area
    criteria, allowing for more focused analysis on relevant segments.

    The returned annotations is processed and used for visualization.
    """
    # ===================== Operation ==========================================

    # Load image
    filepath = str(DATA_DIR / "images" / "eggs_carton.jpg")
    image = io.load_image(filepath=filepath) 
    logger.success(f"Loaded image from {filepath}")


    # Generate superpixels first
    result_felzenszwalb = cornea.segment_image_using_felzenszwalb(
        image=image,
        scale=500,
        sigma=1,
        min_size=200
    )

    superpixel_labels = result_felzenszwalb.to_dict()['labeled_mask']

    # Filter superpixels based on area
    result = cornea.filter_segments_by_area(
        image=image,
        labels=superpixel_labels,
        min_area=10000,
        max_area=100000
    )

    # Access results
    annotation = result.to_dict()
    logger.success("Filtering completed.")

    # ===================== Visualization  (Optional) ======================

    # Initialize Rerun for visualization
    rr.init("cornea_filter_segmentation_by_area", spawn=True)
    rr.send_blueprint(
        rrb.Blueprint(
            rrb.Grid(
                rrb.Spatial2DView(name="Original", origin="input"),
                rrb.Spatial2DView(name="Filtered Mask", origin="segmented_mask"),
            ),
            rrb.SelectionPanel(),
            rrb.TimePanel(),
        ),
        make_active=True,
    )

    # Input Image
    in_np_image = image.to_numpy()

    # Filtered Labeled mask
    filtered_labels_np = annotation['labeled_mask']

    # Log Images
    rr.log("input", rr.Image(in_np_image))
    rr.log("segmented_mask", rr.SegmentationImage(filtered_labels_np))


def filter_segments_by_color_example():
    """
    Filters superpixels based on color.

    This function filters out superpixels that do not meet the specified color
    criteria, allowing for more focused analysis on relevant segments.

    The returned annotations is processed and used for visualization.
    """
    # ===================== Operation ==========================================

    # Load image
    filepath = str(DATA_DIR / "images" / "eggs_carton.jpg")
    image = io.load_image(filepath=filepath) 
    logger.success(f"Loaded image from {filepath}")

    
    # Generate superpixels first
    result_felzenszwalb = cornea.segment_image_using_felzenszwalb(
        image=image,
        scale=500,
        sigma=1,
        min_size=200
    )

    superpixel_labels = result_felzenszwalb.to_dict()['labeled_mask']
    superpixel_labels = datatypes.Image(superpixel_labels)

    # Filter superpixels based on color
    result = cornea.filter_segments_by_color(
        image=image,
        labels=superpixel_labels,
        min_color=0,
        max_color=125.0
    )

    # Access results
    annotation = result.to_dict()
    logger.success("Filtering completed.")

    # ===================== Visualization  (Optional) ======================

    # Initialize Rerun for visualization
    rr.init("cornea_filter_segmentation_by_color", spawn=True)
    rr.send_blueprint(
        rrb.Blueprint(
            rrb.Grid(
                rrb.Spatial2DView(name="Original", origin="input"),
                rrb.Spatial2DView(name="Filtered Mask", origin="segmented_mask"),
            ),
            rrb.SelectionPanel(),
            rrb.TimePanel(),
        ),
        make_active=True,
    )

    # Input Image
    in_np_image = image.to_numpy()

    # Filtered Labeled mask
    filtered_labels_np = annotation['labeled_mask']

    # Log Images
    rr.log("input", rr.Image(in_np_image))
    rr.log("segmented_mask", rr.SegmentationImage(filtered_labels_np))


def filter_segments_by_mask_example():
    """
    Filters superpixels based on a mask.

    This function filters out superpixels that do not intersect with the given mask,
    allowing for more focused analysis on relevant segments.

    The returned annotations is processed and used for visualization.
    """
    # ===================== Operation ==========================================

    # Load and preprocess image
    filepath = str(DATA_DIR / "images" / "eggs_carton.jpg")
    image = io.load_image(filepath=filepath) 
    logger.success(f"Loaded image from {filepath}")
    
    # Generate superpixels first
    result_felzenszwalb = cornea.segment_image_using_felzenszwalb(
        image=image,
        scale=500,
        sigma=1,
        min_size=200
    )

    superpixel_labels = result_felzenszwalb.to_dict()['labeled_mask']
    superpixel_labels = datatypes.Image(superpixel_labels)

    # Create mask as one third of the image size
    def half_mask(image):
        h, w, _ = image.shape
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[:, :w//3] = 1
        return mask
    
    mask = half_mask(image.to_numpy()) * 255
    mask = datatypes.Image(image=mask, color_model='L')

    # Apply filtering
    result = cornea.filter_segments_by_mask(
        image=image,
        labels=superpixel_labels,
        mask=mask
    )

    # Access results
    annotation = result.to_dict()
    logger.success("Filtering completed.")

    # ===================== Visualization  (Optional) ======================

    # Initialize Rerun for visualization
    rr.init("cornea_filter_segmentation_by_mask", spawn=True)
    rr.send_blueprint(
        rrb.Blueprint(
            rrb.Grid(
                rrb.Spatial2DView(name="Input", origin="input"),
                rrb.Spatial2DView(name="Filtering Mask", origin="input_mask"),
                rrb.Spatial2DView(name="Filtered Mask", origin="segmented_mask"),
            ),
            rrb.SelectionPanel(),
            rrb.TimePanel(),
        ),
        make_active=True,
    )

    # Input image
    in_np_image = image.to_numpy()

    # Mask
    mask_np_in = mask.to_numpy()

    # Filtered
    filtered_labels_np = annotation['labeled_mask']

    # Log images
    rr.log("input", rr.Image(in_np_image))
    rr.log("input_mask", rr.SegmentationImage(mask_np_in))
    rr.log("segmented_mask", rr.Image(filtered_labels_np))


# ===================== Graph Examples =====================

def segment_image_using_grab_cut_example():
    """
    Performs GrabCut segmentation.

    This function applies the GrabCut algorithm to separate a foreground object
    from the background, given an initial bounding box or mask. The result can be
    used for object extraction in robotic perception and vision-based inspection.

    The returned annotations is processed and used for visualization.
    """
    # ===================== Operation ==========================================
    
    # Load image
    filepath = str(DATA_DIR / "images" / "plastic_part.jpg")
    image = io.load_image(filepath=filepath)
    logger.success(f"Loaded image from {filepath}")

    # Define bounding box
    bbox = [220, 20, 930, 850]  # [x, y, width, height] top left x and y
    bbox_dt = datatypes.Boxes2D(arrays=bbox, array_format='XYWH')

    # Segment image
    annotation = cornea.segment_image_using_grab_cut(
        image=image,
        num_iterations=2,
        bbox=bbox_dt
    )
    
    # Access results
    annotation = annotation.to_dict()
    logger.success("Segmentation completed.")

    # ===================== Visualization  (Optional) ======================

    # Initialize Rerun for visualization
    rr.init("cornea_grab_cut_segmentation", spawn=True)
    rr.send_blueprint(
        rrb.Blueprint(
            rrb.Horizontal(
                rrb.Spatial2DView(name="Image", origin="input"),
                rrb.Spatial2DView(name="Mask", origin="segmented_mask"),
            ),
            rrb.SelectionPanel(),
            rrb.TimePanel(),
        ),
        make_active=True,
    )

    # Input image
    in_np_image = image.to_numpy()

    # Segmentation mask
    mask_np = annotation['labeled_mask']

    # Log input image overlaid with bbox
    rr.log("input", rr.Image(in_np_image))
    rr.log("input", rr.Boxes2D(
        array=np.array([bbox]),
        array_format=rr.Box2DFormat.XYWH,
        colors=[0, 255, 0],
    ))

    rr.log("segmented_mask", rr.Image(mask_np))


# ===================== Deep Learning Examples =====================

def segment_image_using_foreground_birefnet_example():
    """
    Segments the foreground using BiRefNet.

    This function applies a pretrained BiRefNet foreground extractor to separate
    the primary object from the background. It is suitable for object isolation
    in robotic perception, bin picking, and vision-based inspection where a clear
    foreground is required.

    The returned annotations is processed and used for visualization.
    """
    # ===================== Operation ==========================================

    # Load image
    filepath = str(DATA_DIR / "images" / "screws_standing.jpg")
    image = io.load_image(filepath=filepath)
    logger.success(f"Loaded image from {filepath}")

    # Perform segmentation
    result = cornea.segment_image_using_foreground_birefnet(
        image=image,
        input_height=1024,
        input_width=1024,
        threshold=0
    )

    # Access results
    annotation = result.to_dict()
    logger.success("Segmentation completed.")

    # ===================== Visualization  (Optional) ======================

    # Initialize Rerun for visualization
    rr.init("cornea_foreground_birefnet_segmentation", spawn=True)
    rr.send_blueprint(
        rrb.Blueprint(
            rrb.Grid(
                rrb.Spatial2DView(name="Original", origin="input"),
                rrb.Spatial2DView(name="Mask", origin="segmented_mask"),
            ),
            rrb.SelectionPanel(),
            rrb.TimePanel(),
        ),
        make_active=True,
    )

    # Input image
    in_np_image = image.to_numpy()

    # Segmentation mask
    mask_np = annotation['labeled_mask']

    # Log images
    rr.log("input", rr.Image(in_np_image))
    rr.log("segmented_mask", rr.Image(mask_np))


def segment_image_using_sam_example():
    """
    Performs segmentation using SAM (Segment Anything Model).

    This function demonstrates how to segment an image using SAM.

    The returned annotations is processed and used for visualization.
    """
    # ===================== Operation ==========================================

    # Load image
    filepath = str(DATA_DIR / "images" / "weld_clamp_0_raw.png")
    image = io.load_image(filepath=filepath)
    logger.success(f"Loaded image from {filepath}")

    # Define bounding boxes for SAM segmentation
    bboxes = [[400, 150, 1200, 450],
              [764, 515, 1564, 815]]
    
    # Segment image
    result = cornea.segment_image_using_sam(
        image=image,
        bboxes=bboxes
    )

    # Access results
    annotations = result.to_list()
    logger.success("Segmentation completed.")

    # ===================== Visualization  (Optional) ======================

    # Initialize Rerun for visualization
    rr.init("cornea_sam_segmentation", spawn=True)
    rr.send_blueprint(
        rrb.Blueprint(
                rrb.Horizontal(
                    rrb.Spatial2DView(name="Input", origin="input"),
                    rrb.Spatial2DView(name="Bboxes & Segments", origin="segmented"),
                ),
            rrb.SelectionPanel(),
            rrb.TimePanel(),
        ),
        make_active=True,
    )

    in_np_image = image.to_numpy()
    h, w = in_np_image.shape[:2]

    # 1. Original image
    rr.log("input", rr.Image(in_np_image))

    # 2. Overlay image with bboxes and segments
    rr.log("segmented/image", rr.Image(in_np_image))  # Log original as base for overlay
    
    # Create empty masks
    h, w = in_np_image.shape[:2]
    masks = []
    masks_with_ids = []
    segmentation_img = np.zeros((h, w), dtype=np.uint16)

    # --- boxes: extract from annotations (preferred) ---
    ann_bboxes = []
    class_ids = []

    for idx, ann in enumerate(annotations):

        label = idx + 1
        mask_i = np.zeros((h, w), dtype=np.uint8)

        if "mask" in ann and isinstance(ann["mask"], np.ndarray):
            m = ann["mask"]
            if m.dtype.kind in ("f", "b"):
                mask_i = (m > 0.5).astype(np.uint8)
            else:
                mask_i = (m > 0).astype(np.uint8)

        elif "segmentation" in ann and ann["segmentation"]:
            seg = ann["segmentation"]
            if isinstance(seg, dict):
                mask_dec = mask_utils.decode(seg)
                if mask_dec.ndim == 3:
                    mask_dec = mask_dec[:, :, 0]
                mask_i = (mask_dec > 0).astype(np.uint8)
            elif isinstance(seg, list) and len(seg) > 0:
                temp = np.zeros((h, w), dtype=np.uint8)
                polys = seg if isinstance(seg[0], list) else [seg]
                for poly in polys:
                    pts = np.array(poly).reshape(-1, 2).astype(np.int32)
                    cv2.fillPoly(temp, [pts], 1)
                mask_i = (temp > 0).astype(np.uint8)

        if mask_i.sum() == 0:
            logger.info(f"Skipping annotation {idx} with label {label} due to empty mask.")
            continue

        masks.append(mask_i)
        masks_with_ids.append((label, mask_i))
        segmentation_img[mask_i > 0] = label

        bbox = ann.get("bbox", None)
        if bbox is None:
            continue

        ann_bboxes.append(list(bbox))
        class_ids.append(label)

    # Log segmentation masks
    rr.log("segmented/masks", rr.SegmentationImage(segmentation_img))

    # Log bounding boxes
    if ann_bboxes:
        rr.log(
            "segmented/boxes",
            rr.Boxes2D(
                array=np.asarray(ann_bboxes, dtype=np.float32),
                array_format=rr.Box2DFormat.XYWH,
                class_ids=np.asarray(class_ids, dtype=np.int32),
            ),
        )


def get_example_dict():
    """Returns a dictionary mapping SDK function names to their example functions."""
    return {
        # Color Examples
        "segment_image_using_rgb": segment_image_using_rgb_example,  
        "segment_image_using_hsv": segment_image_using_hsv_example,  
        "segment_image_using_lab": segment_image_using_lab_example,  
        "segment_image_using_ycrcb": segment_image_using_ycrcb_example, 

        # Region Examples
        "segment_image_using_focus_region": segment_image_using_focus_region_example,  
        "segment_image_using_watershed": segment_image_using_watershed_example,  
        "segment_image_using_flood_fill": segment_image_using_flood_fill_example,  

        # Deep Learning Examples
        "segment_image_using_foreground_birefnet": segment_image_using_foreground_birefnet_example,  
        "segment_image_using_sam": segment_image_using_sam_example, 

        # Graph Examples
        "segment_image_using_grab_cut": segment_image_using_grab_cut_example, 

        # Superpixel Examples
        "segment_image_using_felzenszwalb": segment_image_using_felzenszwalb_example,  
        "segment_image_using_slic_superpixel": segment_image_using_slic_superpixel_example, 
        "filter_segments_by_area": filter_segments_by_area_example,  
        "filter_segments_by_color": filter_segments_by_color_example, 
        "filter_segments_by_mask": filter_segments_by_mask_example,  

        # Threshold Examples
        "segment_image_using_otsu_threshold": segment_image_using_otsu_threshold_example,  
        "segment_image_using_local_threshold": segment_image_using_local_threshold_example,  
        "segment_image_using_yen_threshold": segment_image_using_yen_threshold_example,  
        "segment_image_using_threshold": segment_image_using_threshold_example,  
        "segment_image_using_adaptive_threshold": segment_image_using_adaptive_threshold_example,  
        "segment_image_using_laplacian_threshold": segment_image_using_laplacian_threshold_example,  

 
    }


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run cornea examples")
    parser.add_argument(
        "--example",
        type=str,
        default="segment_image_using_watershed",   # <-- default example name
        help="Name of the example to run (without _example suffix) or use --list to see all available examples",
    )
    parser.add_argument(
        "--list", action="store_true", help="List all available examples"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    example_dict = get_example_dict()

    if args.list:
        logger.info("Available examples:")
        for example_name in sorted(example_dict.keys()):
            logger.info(f" - {example_name}")
        return

    if not args.example:
        logger.error(
            "Please provide --example or use --list to see all available examples."
        )
        raise SystemExit(1)

    args.example = args.example.lower()

    if args.example not in example_dict:
        logger.error(f"Example '{args.example}' not found.")

        # Find the one that is nearest in name
        close_matches = difflib.get_close_matches(
            args.example, example_dict.keys(), n=3, cutoff=0.4
        )

        if close_matches:
            logger.error(f"Did you mean one of these?")
            for match in close_matches:
                logger.error(f"  - {match}")

        raise SystemExit(1)

    logger.info(f"Running {args.example} example...")
    example_dict[args.example]()
    logger.success(f"Example {args.example} completed.")


if __name__ == "__main__":
    main()
