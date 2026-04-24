"""
This example demonstrates how to use the Pupil SDK for image processing operations, including contrast enhancement,
morphological transformations, pyramid transformations, ridge/vesselness filters, and sharpening/gradient filters.

It also shows how to visualize results using Rerun.
"""

import argparse
import difflib
import pathlib
import numpy as np

from loguru import logger
import rerun as rr
import rerun.blueprint as rrb

from telekinesis import pupil
from datatypes import io

ROOT_DIR = pathlib.Path(__file__).parent.parent.resolve()
DATA_DIR = ROOT_DIR / "telekinesis-data"


# ===================== Morphology =====================


def filter_image_using_morphological_erode_example():
    """
    Applies erosion to shrink bright regions and remove small noise.

    Erosion removes pixels from object boundaries, useful for removing
    small bright spots and shrinking objects.

    The returned image is processed and used for visualization.
    """
    # ===================== Operation ==========================================

    # Load image
    filepath = str(DATA_DIR / "images" / "gear_with_texture.jpg")
    image = io.load_image(
        filepath=filepath, as_binary=True, binary_method="otsu"
    )
    logger.success(f"Loaded image from {filepath}")

    # Apply erosion morphological operation
    filtered_image = pupil.filter_image_using_morphological_erode(
        image=image,
        kernel_size=5,
        kernel_shape="ellipse",
        iterations=10,
        border_type="default",
    )

    # Access results
    filtered_image_np = filtered_image.to_numpy()
    logger.success(
        "Applied erosion morphological operation. Output image shape: {}",
        filtered_image_np.shape,
    )

    # ===================== Visualization  (Optional) ======================

    # Initialize Rerun for visualization
    rr.init("filter_image_using_morphological_erode", spawn=True)
    rr.send_blueprint(
        rrb.Blueprint(
            rrb.Grid(
                rrb.Spatial2DView(name="Original", origin="input"),
                rrb.Spatial2DView(name="Filtered", origin="filtered_image"),
            ),
            rrb.SelectionPanel(),
            rrb.TimePanel(),
        ),
        make_active=True,
    )

    # Input image
    image_np = image.to_numpy()

    # Log images
    rr.log("input", rr.Image(image_np))
    rr.log("filtered_image", rr.Image(filtered_image_np))


def filter_image_using_morphological_dilate_example():
    """
    Applies dilation to expand bright regions and fill holes.

    Dilation adds pixels to object boundaries, useful for filling gaps
    and expanding objects.

    The returned image is processed and used for visualization.
    """
    # ===================== Operation ==========================================

    # Load image
    filepath = str(DATA_DIR / "images" / "spanners_arranged.jpg")
    image = io.load_image(
        filepath=filepath, as_binary=True, binary_method="otsu"
    )
    logger.success(f"Loaded image from {filepath}")

    # Apply dilation morphological operation
    filtered_image = pupil.filter_image_using_morphological_dilate(
        image=image,
        kernel_size=5,
        kernel_shape="ellipse",
        iterations=5,
        border_type="constant",
        border_value=0,
    )

    # Access results
    filtered_image_np = filtered_image.to_numpy()
    logger.success(
        "Applied dilation morphological operation. Output image shape: {}",
        filtered_image_np.shape,
    )

    # ===================== Visualization  (Optional) ======================

    # Initialize Rerun for visualization
    rr.init("filter_image_using_morphological_dilate", spawn=True)
    rr.send_blueprint(
        rrb.Blueprint(
            rrb.Grid(
                rrb.Spatial2DView(name="Original", origin="input"),
                rrb.Spatial2DView(name="Filtered", origin="filtered_image"),
            ),
            rrb.SelectionPanel(),
            rrb.TimePanel(),
        ),
        make_active=True,
    )

    # Input image
    image_np = image.to_numpy()

    # Log images
    rr.log("input", rr.Image(image_np))
    rr.log("filtered_image", rr.Image(filtered_image_np))


def filter_image_using_morphological_close_example():
    """
    Applies a close morphological transformation.

    Closing consists of a dilation followed by an erosion. It is useful for
    filling small holes, closing gaps between nearby objects, and smoothing
    object boundaries.

    The returned image is processed and used for visualization.
    """
    # ===================== Operation ==========================================

    # Load image
    filepath = str(DATA_DIR / "images" / "nuts_scattered.jpg")
    image = io.load_image(filepath=filepath, as_binary=True)
    logger.success(f"Loaded image from {filepath}")

    # Apply close morphological operation
    filtered_image = pupil.filter_image_using_morphological_close(
        image=image,
        kernel_size=5,
        kernel_shape="ellipse",
        iterations=5,
        border_type="default",
    )

    # Access results
    filtered_image_np = filtered_image.to_numpy()
    logger.success(
        "Applied close morphological operation. Output image shape: {}",
        filtered_image_np.shape,
    )

    # ===================== Visualization  (Optional) ======================

    # Initialize Rerun for visualization
    rr.init("filter_image_using_morphological_close", spawn=True)
    rr.send_blueprint(
        rrb.Blueprint(
            rrb.Grid(
                rrb.Spatial2DView(name="Original", origin="input"),
                rrb.Spatial2DView(name="Filtered", origin="filtered_image"),
            ),
            rrb.SelectionPanel(),
            rrb.TimePanel(),
        ),
        make_active=True,
    )

    # Input image
    image_np = image.to_numpy()

    # Log images
    rr.log("input", rr.Image(image_np))
    rr.log("filtered_image", rr.Image(filtered_image_np))


def filter_image_using_morphological_open_example():
    """
    Applies a open morphological transformation.

    Opening consists of an erosion followed by a dilation. It is useful for
    removing small bright noise, separating weakly connected objects, and
    smoothing object boundaries without significantly altering the overall shape.

    The returned image is processed and used for visualization.
    """
    # ===================== Operation ==========================================

    # Load image
    filepath = str(DATA_DIR / "images" / "broken_cables.png")
    image = io.load_image(
        filepath=filepath, as_binary=True, binary_method="fixed"
    )
    logger.success(f"Loaded image from {filepath}")

    # Apply open morphological operation
    filtered_image = pupil.filter_image_using_morphological_open(
        image=image,
        kernel_size=3,
        kernel_shape="ellipse",
        iterations=2,
        border_type="constant",
        border_value=0,
    )

    # Access results
    filtered_image_np = filtered_image.to_numpy()
    logger.success(
        "Applied open morphological operation. Output image shape: {}",
        filtered_image_np.shape,
    )

    # ===================== Visualization  (Optional) ======================

    # Initialize Rerun for visualization
    rr.init("filter_image_using_morphological_open", spawn=True)
    rr.send_blueprint(
        rrb.Blueprint(
            rrb.Grid(
                rrb.Spatial2DView(name="Original", origin="input"),
                rrb.Spatial2DView(name="Filtered", origin="filtered_image"),
            ),
            rrb.SelectionPanel(),
            rrb.TimePanel(),
        ),
        make_active=True,
    )

    # Input image
    image_np = image.to_numpy()

    # Log images
    rr.log("input", rr.Image(image_np))
    rr.log("filtered_image", rr.Image(filtered_image_np))


def filter_image_using_morphological_gradient_example():
    """
    Applies a gradient morphological transformation.

    The morphological gradient computes the difference between dilation and
    erosion of an image. It highlights object boundaries by emphasizing the
    intensity transitions at the edges.

    The returned image is processed and used for visualization.
    """
    # ===================== Operation ==========================================

    # Load image
    filepath = str(DATA_DIR / "images" / "cartons_arranged.png")
    image = io.load_image(filepath=filepath, as_gray=True)
    logger.success(f"Loaded image from {filepath}")

    # Apply gradient morphological operation
    filtered_image = pupil.filter_image_using_morphological_gradient(
        image=image,
        kernel_size=5,
        kernel_shape="ellipse",
        iterations=1,
        border_type="default",
    )

    # Access results
    filtered_image_np = filtered_image.to_numpy()
    logger.success(
        "Applied gradient morphological operation. Output image shape: {}",
        filtered_image_np.shape,
    )

    # ===================== Visualization  (Optional) ======================

    # Initialize Rerun for visualization
    rr.init("filter_image_using_morphological_gradient", spawn=True)
    rr.send_blueprint(
        rrb.Blueprint(
            rrb.Grid(
                rrb.Spatial2DView(name="Original", origin="input"),
                rrb.Spatial2DView(name="Filtered", origin="filtered_image"),
            ),
            rrb.SelectionPanel(),
            rrb.TimePanel(),
        ),
        make_active=True,
    )

    # Input image
    image_np = image.to_numpy()

    # Log images
    rr.log("input", rr.Image(image_np))
    rr.log("filtered_image", rr.Image(filtered_image_np))


def filter_image_using_morphological_tophat_example():
    """
    Applies a tophat morphological transformation.

    The top-hat transform computes the difference between the original image
    and its morphological opening. It highlights small bright objects or
    features that are smaller than the structuring element.

    The returned image is processed and used for visualization.
    """
    # ===================== Operation ==========================================

    # Load image
    filepath = str(DATA_DIR / "images" / "keyhole.jpg")
    image = io.load_image(filepath=filepath, as_gray=True)
    logger.success(f"Loaded image from {filepath}")

    # Apply tophat morphological operation
    filtered_image = pupil.filter_image_using_morphological_tophat(
        image=image,
        kernel_size=3,
        kernel_shape="ellipse",
        iterations=5,
        border_type="default",
    )

    # Access results
    filtered_image_np = filtered_image.to_numpy()
    logger.success(
        "Applied tophat morphological operation. Output image shape: {}",
        filtered_image_np.shape,
    )

    # ===================== Visualization  (Optional) ======================

    # Initialize Rerun for visualization
    rr.init("filter_image_using_morphological_tophat", spawn=True)
    rr.send_blueprint(
        rrb.Blueprint(
            rrb.Grid(
                rrb.Spatial2DView(name="Original", origin="input"),
                rrb.Spatial2DView(name="Filtered", origin="filtered_image"),
            ),
            rrb.SelectionPanel(),
            rrb.TimePanel(),
        ),
        make_active=True,
    )

    # Input image
    image_np = image.to_numpy()

    # Log images
    rr.log("input", rr.Image(image_np))
    rr.log("filtered_image", rr.Image(filtered_image_np))


def filter_image_using_morphological_blackhat_example():
    """
    Applies a blackhat morphological transformation.

    This operation is useful for enhancing dark details, detecting small
    dark objects on bright backgrounds, and correcting uneven illumination.

    The returned image is processed and used for visualization.
    """
    # ===================== Operation ==========================================

    # Load image
    filepath = str(DATA_DIR / "images" / "mechanical_parts_gray.png")
    image = io.load_image(filepath=filepath, as_gray=True)
    logger.success(f"Loaded image from {filepath}")

    # Apply blackhat morphological operation
    filtered_image = pupil.filter_image_using_morphological_blackhat(
        image=image,
        kernel_size=15,
        kernel_shape="ellipse",
        iterations=2,
        border_type="default",
    )

    # Access results
    filtered_image_np = filtered_image.to_numpy()
    logger.success(
        "Applied blackhat morphological operation. Output image shape: {}",
        filtered_image_np.shape,
    )

    # ===================== Visualization  (Optional) ======================

    # Initialize Rerun for visualization
    rr.init("filter_image_using_morphological_blackhat", spawn=True)
    rr.send_blueprint(
        rrb.Blueprint(
            rrb.Grid(
                rrb.Spatial2DView(name="Original", origin="input"),
                rrb.Spatial2DView(name="Filtered", origin="filtered_image"),
            ),
            rrb.SelectionPanel(),
            rrb.TimePanel(),
        ),
        make_active=True,
    )

    # Input image
    image_np = image.to_numpy()

    # Log images
    rr.log("input", rr.Image(image_np))
    rr.log("filtered_image", rr.Image(filtered_image_np))


def filter_image_using_morphological_hitmiss_example():
    """
    Applies a hit-miss morphological transformation.

    Hit-miss detects specific patterns in binary images by matching foreground
    and background pixels simultaneously. Useful for corner detection, thinning,
    and finding custom shapes.

    The returned image is processed and used for visualization.
    """
    # ===================== Operation ==========================================

    # Load image
    filepath = str(DATA_DIR / "images" / "spanners_arranged.jpg")
    image = io.load_image(
        filepath=filepath, as_binary=True, binary_method="fixed"
    )
    logger.success(f"Loaded image from {filepath}")

    # Apply hit-miss morphological operation
    filtered_image = pupil.filter_image_using_morphological_hitmiss(
        image=image,
        kernel_size=5,
        kernel_shape="ellipse",
        iterations=1,
        border_type="constant",
        border_value=0,
    )

    # Access results
    filtered_image_np = filtered_image.to_numpy()
    logger.success(
        "Applied hit-miss morphological operation. Output image shape: {}",
        filtered_image_np.shape,
    )

    # ===================== Visualization  (Optional) ======================

    # Initialize Rerun for visualization
    rr.init("filter_image_using_morphological_hitmiss", spawn=True)
    rr.send_blueprint(
        rrb.Blueprint(
            rrb.Grid(
                rrb.Spatial2DView(name="Original", origin="input"),
                rrb.Spatial2DView(name="Filtered", origin="filtered_image"),
            ),
            rrb.SelectionPanel(),
            rrb.TimePanel(),
        ),
        make_active=True,
    )

    # Input image
    image_np = image.to_numpy()

    # Log images
    rr.log("input", rr.Image(image_np))
    rr.log("filtered_image", rr.Image(filtered_image_np))


# ===================== Ridge / Vesselness =====================


def filter_image_using_frangi_example():
    """
    Applies Frangi vesselness filter to enhance tubular structures.

    Frangi filter is designed to detect vessel-like structures in medical
    images, fingerprints, and other images with elongated features.

    The returned image is processed and used for visualization.
    """
    # ===================== Operation ==========================================
    # Load image
    filepath = str(DATA_DIR / "images" / "tablets_arranged.jpg")
    image = io.load_image(filepath=filepath, as_gray=True)
    logger.success(f"Loaded image from {filepath}")

    # Apply Frangi filter for vesselness enhancement
    filtered_image = pupil.filter_image_using_frangi(
        image=image,
        scale_start=6,
        scale_end=10,
        scale_step=1,
        alpha=0.5,
        beta=0.5,
        detect_black_ridges=True,
        border_type="reflect",
        border_value=0.0,
    )

    # Access results
    filtered_image_np = filtered_image.to_numpy()
    logger.success(
        "Applied Frangi filter. Filtered output image shape: {}",
        filtered_image_np.shape,
    )

    # ===================== Visualization  (Optional) ======================

    # Initialize Rerun for visualization
    rr.init("filter_image_using_frangi", spawn=True)
    rr.send_blueprint(
        rrb.Blueprint(
            rrb.Grid(
                rrb.Spatial2DView(name="Original", origin="input"),
                rrb.Spatial2DView(name="Filtered", origin="filtered_image"),
            ),
            rrb.SelectionPanel(),
            rrb.TimePanel(),
        ),
        make_active=True,
    )

    # Input image
    image_np = image.to_numpy()

    # Log images
    rr.log("input", rr.Image(image_np))
    rr.log("filtered_image", rr.Image(filtered_image_np))


def filter_image_using_hessian_example():
    """
    Applies Hessian-based vesselness filter for tubular structure detection.

    Hessian filter uses eigenvalue analysis to detect vessel-like structures,
    similar to Frangi but with different vesselness measure.

    The returned image is processed and used for visualization.
    """
    # ===================== Operation ==========================================
    # Load image
    filepath = str(DATA_DIR / "images" / "wires.jpg")
    image = io.load_image(filepath=filepath, as_gray=True)
    logger.success(f"Loaded image from {filepath}")

    # Apply Hessian-based vesselness filter
    filtered_image = pupil.filter_image_using_hessian(
        image=image,
        scale_start=1,
        scale_end=6,
        scale_step=1,
        alpha=0.5,
        beta=0.5,
        gamma=15,
        detect_black_ridges=True,
        border_type="reflect",
        border_value=0.0,
    )

    # Access results
    filtered_image_np = filtered_image.to_numpy()
    logger.success(
        "Applied Hessian filter. Filtered output image shape: {}",
        filtered_image_np.shape,
    )

    # ===================== Visualization  (Optional) ======================

    # Initialize Rerun for visualization
    rr.init("filter_image_using_hessian", spawn=True)
    rr.send_blueprint(
        rrb.Blueprint(
            rrb.Grid(
                rrb.Spatial2DView(name="Original", origin="input"),
                rrb.Spatial2DView(name="Filtered", origin="filtered_image"),
            ),
            rrb.SelectionPanel(),
            rrb.TimePanel(),
        ),
        make_active=True,
    )

    # Input image
    image_np = image.to_numpy()

    # Log images
    rr.log("input", rr.Image(image_np))
    rr.log("filtered_image", rr.Image(filtered_image_np))


def filter_image_using_sato_example():
    """
    Applies Sato filter for multi-scale ridge detection.

    Sato filter is designed to detect ridges and valleys at multiple scales,
    useful for detecting fine structures like vessels or fibers.

    The returned image is processed and used for visualization.
    """
    # ===================== Operation ==========================================

    # Load image
    filepath = str(DATA_DIR / "images" / "pcb_top_gray.png")
    image = io.load_image(filepath=filepath, as_gray=True)
    logger.success(f"Loaded image from {filepath}")

    # Apply Sato filter for multi-scale ridge detection
    filtered_image = pupil.filter_image_using_sato(
        image=image,
        scale_start=1,
        scale_end=12,
        scale_step=1,
        detect_black_ridges=False,
        border_type="reflect",
        border_value=0.0,
    )

    # Access results
    filtered_image_np = filtered_image.to_numpy()
    logger.success(
        "Applied Sato filter. Filtered output image shape: {}",
        filtered_image_np.shape,
    )

    # ===================== Visualization  (Optional) ======================

    # Initialize Rerun for visualization
    rr.init("filter_image_using_sato", spawn=True)
    rr.send_blueprint(
        rrb.Blueprint(
            rrb.Grid(
                rrb.Spatial2DView(name="Original", origin="input"),
                rrb.Spatial2DView(name="Filtered", origin="filtered_image"),
            ),
            rrb.SelectionPanel(),
            rrb.TimePanel(),
        ),
        make_active=True,
    )

    # Input image
    image_np = image.to_numpy()

    # Log images
    rr.log("input", rr.Image(image_np))
    rr.log("filtered_image", rr.Image(filtered_image_np))


def filter_image_using_meijering_example():
    """
    Applies Meijering filter for neurite detection.

    Meijering filter is optimized for detecting neurites and similar
    branching structures in biomedical images.

    The returned image is processed and used for visualization.
    """
    # ===================== Operation ==========================================

    # Load image
    filepath = str(DATA_DIR / "images" / "sidewalk_cracked.jpg")
    image = io.load_image(filepath=filepath, as_gray=True)
    logger.success(f"Loaded image from {filepath}")

    # Apply Meijering filter for neurite detection
    filtered_image = pupil.filter_image_using_meijering(
        image=image,
        scale_start=1,
        scale_end=10,
        scale_step=2,
        detect_black_ridges=True,
        border_type="reflect",
        border_value=0.0,
    )

    # Access results
    filtered_image_np = filtered_image.to_numpy()
    logger.success(
        "Applied Meijering filter. Filtered output image shape: {}",
        filtered_image_np.shape,
    )

    # ===================== Visualization  (Optional) ======================

    # Initialize Rerun for visualization
    rr.init("filter_image_using_meijering", spawn=True)
    rr.send_blueprint(
        rrb.Blueprint(
            rrb.Grid(
                rrb.Spatial2DView(name="Original", origin="input"),
                rrb.Spatial2DView(name="Filtered", origin="filtered_image"),
            ),
            rrb.SelectionPanel(),
            rrb.TimePanel(),
        ),
        make_active=True,
    )

    # Input image
    image_np = image.to_numpy()

    # Log images
    rr.log("input", rr.Image(image_np))
    rr.log("filtered_image", rr.Image(filtered_image_np))


# ===================== Sharpening / Gradients =====================


def filter_image_using_laplacian_example():
    """
    Applies Laplacian filter for edge detection using second derivatives.

    Laplacian operator detects edges by finding regions where the second
    derivative is zero or changes sign.

    The returned image is processed and used for visualization.
    """
    # ===================== Operation ==========================================

    # Load image
    filepath = str(
        DATA_DIR / "images" / "flat_mechanical_component_denoised.png"
    )
    image = io.load_image(filepath=filepath)
    logger.success(f"Loaded image from {filepath}")

    # Apply Laplacian filter for edge detection
    filtered_image = pupil.filter_image_using_laplacian(
        image=image,
        output_format="32bit",
        kernel_size=5,
        scale=1.0,
        delta=0.0,
        border_type="default",
    )

    # Access results
    filtered_image_np = filtered_image.to_numpy()
    logger.debug(
        "Laplacian filter output image dtype: {}", filtered_image_np.dtype
    )
    logger.success(
        "Applied Laplacian filter. Filtered output image shape: {}",
        filtered_image_np.shape,
    )

    # ===================== Visualization  (Optional) ======================

    # Handle float images for visualization
    filtered_image_np = np.abs(filtered_image_np.astype(np.float32))
    filtered_image_np = np.clip(filtered_image_np, 0, 255)
    filtered_image_np = filtered_image_np.astype(np.uint8)

    # Initialize Rerun for visualization
    rr.init("filter_image_using_laplacian", spawn=True)
    rr.send_blueprint(
        rrb.Blueprint(
            rrb.Grid(
                rrb.Spatial2DView(name="Original", origin="input"),
                rrb.Spatial2DView(name="Filtered", origin="filtered_image"),
            ),
            rrb.SelectionPanel(),
            rrb.TimePanel(),
        ),
        make_active=True,
    )

    # Input image
    image_np = image.to_numpy()

    # Log images
    rr.log("input", rr.Image(image_np))
    rr.log("filtered_image", rr.Image(filtered_image_np))


def filter_image_using_sobel_example():
    """
    Applies Sobel filter for directional edge detection.

    Sobel operator computes gradients in X and Y directions, useful for
    detecting edges and their orientation.

    The returned image is processed and used for visualization.
    """
    # ===================== Operation ==========================================

    # Load image
    filepath = str(DATA_DIR / "images" / "nuts.jpg")
    image = io.load_image(filepath=filepath, as_gray=True)
    logger.success(f"Loaded image from {filepath}")

    # Apply Sobel filter for edge detection
    filtered_image = pupil.filter_image_using_sobel(
        image=image,
        output_format="64bit",
        dx=1,
        dy=1,
        kernel_size=9,
        scale=1.0,
        delta=0.0,
        border_type="default",
    )

    # Access results
    filtered_image_np = filtered_image.to_numpy()
    logger.success(
        "Applied Sobel filter. Filtered output image shape: {}",
        filtered_image_np.shape,
    )

    # ===================== Visualization (Optional) ======================

    # Handle float images for visualization
    filtered_image_np = np.abs(filtered_image_np.astype(np.float32))
    filtered_image_np = np.clip(filtered_image_np, 0, 255)
    filtered_image_np = filtered_image_np.astype(np.uint8)

    # Initialize Rerun for visualization
    rr.init("filter_image_using_sobel", spawn=True)
    rr.send_blueprint(
        rrb.Blueprint(
            rrb.Grid(
                rrb.Spatial2DView(name="Original", origin="input"),
                rrb.Spatial2DView(name="Filtered", origin="filtered_image"),
            ),
            rrb.SelectionPanel(),
            rrb.TimePanel(),
        ),
        make_active=True,
    )

    # Input image
    image_np = image.to_numpy()

    # Log images
    rr.log("input", rr.Image(image_np))
    rr.log("filtered_image", rr.Image(filtered_image_np))


def filter_image_using_scharr_example():
    """
    Applies Scharr filter for improved edge detection accuracy.

    Scharr operator is similar to Sobel but with better rotation invariance
    and more accurate gradient estimation.

    The returned image is processed and used for visualization.
    """
    # ===================== Operation ==========================================

    # Load image
    filepath = str(DATA_DIR / "images" / "nuts_scattered.jpg")
    image = io.load_image(filepath=filepath, as_gray=True)
    logger.success(f"Loaded image from {filepath}")

    # Apply Scharr filter for edge detection
    filtered_image = pupil.filter_image_using_scharr(
        image=image,
        output_format="same as input",
        dx=0,
        dy=1,
        scale=1.0,
        delta=0.0,
        border_type="default",
    )

    # Access results
    filtered_image_np = filtered_image.to_numpy()
    logger.success(
        "Applied Scharr filter. Filtered output image shape: {}",
        filtered_image_np.shape,
    )

    # ===================== Visualization  (Optional) ======================

    # Initialize Rerun for visualization
    rr.init("filter_image_using_scharr", spawn=True)
    rr.send_blueprint(
        rrb.Blueprint(
            rrb.Grid(
                rrb.Spatial2DView(name="Original", origin="input"),
                rrb.Spatial2DView(name="Filtered", origin="filtered_image"),
            ),
            rrb.SelectionPanel(),
            rrb.TimePanel(),
        ),
        make_active=True,
    )

    # Input image
    image_np = image.to_numpy()

    # Log images
    rr.log("input", rr.Image(image_np))
    rr.log("filtered_image", rr.Image(filtered_image_np))


def filter_image_using_gabor_example():
    """
    Applies Gabor filter for texture analysis and feature detection.

    Gabor filters are useful for detecting oriented features and textures
    at specific scales and orientations.

    The returned image is processed and used for visualization.
    """
    # ===================== Operation ==========================================

    # Load image
    filepath = str(DATA_DIR / "images" / "finger_print.jpg")
    image = io.load_image(filepath=filepath, as_gray=True)
    logger.success(f"Loaded image from {filepath}")

    # Apply Gabor filter for texture analysis and feature detection
    filtered_image = pupil.filter_image_using_gabor(
        image=image,
        kernel_size=5,
        standard_deviation=5.0,
        orientation=90.0,
        wavelength=5.0,
        aspect_ratio=0.5,
        phase_offset=90.0,
        output_format="8bit",
    )

    # Access results
    filtered_image_np = filtered_image.to_numpy()
    logger.success(
        "Applied Gabor filter. Filtered output image shape: {}",
        filtered_image_np.shape,
    )

    # ===================== Visualization  (Optional) ======================

    # Handle float images for visualization
    filtered_image_np = np.abs(filtered_image_np.astype(np.float32))
    min_val = filtered_image_np.min()
    max_val = filtered_image_np.max()
    if max_val > min_val:
        img = (filtered_image_np - min_val) / (max_val - min_val)
    else:
        img = np.zeros_like(filtered_image_np)
    filtered_image_np = (img * 255).astype(np.uint8)

    # Initialize Rerun for visualization
    rr.init("filter_image_using_gabor", spawn=True)
    rr.send_blueprint(
        rrb.Blueprint(
            rrb.Grid(
                rrb.Spatial2DView(name="Original", origin="input"),
                rrb.Spatial2DView(name="Filtered", origin="filtered_image"),
            ),
            rrb.SelectionPanel(),
            rrb.TimePanel(),
        ),
        make_active=True,
    )

    # Input image
    image_np = image.to_numpy()

    # Log images
    rr.log("input", rr.Image(image_np))
    rr.log("filtered_image", rr.Image(filtered_image_np))


# ===================== Smoothing =====================


def filter_image_using_bilateral_example():
    """
    Applies a bilateral filter to reduce noise while preserving edges.

    Bilateral filtering is effective for noise reduction while maintaining
    edge sharpness. It considers both spatial proximity and color similarity.

    The returned image is processed and used for visualization.
    """
    # ===================== Operation ==========================================

    # Load image
    filepath = str(DATA_DIR / "images" / "nuts_scattered_noised.jpg")
    image = io.load_image(filepath=filepath)
    logger.success(f"Loaded image from {filepath}")

    # Apply bilateral filter for edge-preserving smoothing
    filtered_image = pupil.filter_image_using_bilateral(
        image=image,
        neighborhood_diameter=5,
        spatial_sigma=75.0,
        color_intensity_sigma=100.0,
        border_type="default",
    )

    # Access results
    filtered_image_np = filtered_image.to_numpy()
    logger.success(
        "Applied Bilateral filter. Filtered output image shape: {}",
        filtered_image_np.shape,
    )

    # ===================== Visualization  (Optional) ======================

    # Initialize Rerun for visualization
    rr.init("filter_image_using_bilateral", spawn=True)
    rr.send_blueprint(
        rrb.Blueprint(
            rrb.Grid(
                rrb.Spatial2DView(name="Original", origin="input"),
                rrb.Spatial2DView(name="Filtered", origin="filtered_image"),
            ),
            rrb.SelectionPanel(),
            rrb.TimePanel(),
        ),
        make_active=True,
    )

    # Input image
    image_np = image.to_numpy()

    # Log images
    rr.log("input", rr.Image(image_np))
    rr.log("filtered_image", rr.Image(filtered_image_np))


def filter_image_using_box_example():
    """
    Applies a normalized box filter with configurable depth and normalization.

    Box filter performs normalized averaging within a kernel region.
    Useful for basic smoothing operations.

    The returned image is processed and used for visualization.
    """
    # ===================== Operation ==========================================

    # Load image
    filepath = str(DATA_DIR / "images" / "nuts_scattered_noised.jpg")
    image = io.load_image(filepath=filepath)
    logger.success(f"Loaded image from {filepath}")

    # Apply box filter for basic smoothing
    filtered_image = pupil.filter_image_using_box(
        image=image,
        output_format="8bit",
        kernel_size=5,
        normalize=True,
        border_type="reflect",
    )

    # Access results
    filtered_image_np = filtered_image.to_numpy()
    logger.success(
        "Applied Box filter. Filtered output image shape: {}",
        filtered_image_np.shape,
    )

    # ===================== Visualization  (Optional) ======================

    # Initialize Rerun for visualization
    rr.init("filter_image_using_box", spawn=True)
    rr.send_blueprint(
        rrb.Blueprint(
            rrb.Grid(
                rrb.Spatial2DView(name="Original", origin="input"),
                rrb.Spatial2DView(name="Filtered", origin="filtered_image"),
            ),
            rrb.SelectionPanel(),
            rrb.TimePanel(),
        ),
        make_active=True,
    )

    # Input image
    image_np = image.to_numpy()

    # Log images
    rr.log("input", rr.Image(image_np))
    rr.log("filtered_image", rr.Image(filtered_image_np))


def filter_image_using_gaussian_blur_example():
    """
    Applies Gaussian blur for smooth noise reduction.

    Gaussian blur uses a Gaussian kernel for weighted averaging, providing
    natural-looking blur with better edge preservation than simple blur.

    The returned image is processed and used for visualization.
    """
    # ===================== Operation ==========================================

    # Load image
    filepath = str(DATA_DIR / "images" / "nuts_scattered_noised.jpg")
    image = io.load_image(filepath=filepath)
    logger.success(f"Loaded image from {filepath}")

    # Apply Gaussian blur for smooth noise reduction
    filtered_image = pupil.filter_image_using_gaussian_blur(
        image=image,
        kernel_size=19,
        sigma_x=2.0,
        sigma_y=3.0,
        border_type="default",
    )

    # Access results
    filtered_image_np = filtered_image.to_numpy()
    logger.success(
        "Applied Gaussian Blur filter. Filtered output image shape: {}",
        filtered_image_np.shape,
    )

    # ===================== Visualization  (Optional) ======================

    # Initialize Rerun for visualization
    rr.init("filter_image_using_gaussian_blur", spawn=True)
    rr.send_blueprint(
        rrb.Blueprint(
            rrb.Grid(
                rrb.Spatial2DView(name="Original", origin="input"),
                rrb.Spatial2DView(name="Filtered", origin="filtered_image"),
            ),
            rrb.SelectionPanel(),
            rrb.TimePanel(),
        ),
        make_active=True,
    )

    # Input image
    image_np = image.to_numpy()

    # Log images
    rr.log("input", rr.Image(image_np))
    rr.log("filtered_image", rr.Image(filtered_image_np))


def filter_image_using_median_blur_example():
    """
    Applies median blur to reduce salt-and-pepper noise.

    Median blur replaces each pixel with the median of its neighborhood,
    effectively removing impulse noise while preserving edges.

    The returned image is processed and used for visualization.
    """
    # ===================== Operation ==========================================

    # Load image
    filepath = str(DATA_DIR / "images" / "flat_mechanical_component.png")
    image = io.load_image(filepath=filepath)
    logger.success(f"Loaded image from {filepath}")

    # Apply median blur for salt-and-pepper noise reduction
    filtered_image = pupil.filter_image_using_median_blur(
        image=image,
        kernel_size=11,
    )

    # Access results
    filtered_image_np = filtered_image.to_numpy()
    logger.success(
        "Applied Median Blur filter. Filtered output image shape: {}",
        filtered_image_np.shape,
    )

    # ===================== Visualization  (Optional) ======================

    # Initialize Rerun for visualization
    rr.init("filter_image_using_median_blur", spawn=True)
    rr.send_blueprint(
        rrb.Blueprint(
            rrb.Grid(
                rrb.Spatial2DView(name="Original", origin="input"),
                rrb.Spatial2DView(name="Filtered", origin="filtered_image"),
            ),
            rrb.SelectionPanel(),
            rrb.TimePanel(),
        ),
        make_active=True,
    )

    # Input image
    image_np = image.to_numpy()

    # Log images
    rr.log("input", rr.Image(image_np))
    rr.log("filtered_image", rr.Image(filtered_image_np))


def filter_image_using_blur_example():
    """
    Simple average blur.

    The returned image is processed and used for visualization.
    """
    # ===================== Operation ==========================================

    # Load image
    filepath = str(DATA_DIR / "images" / "nuts_scattered_noised.jpg")
    image = io.load_image(filepath=filepath)
    logger.success(f"Loaded image from {filepath}")

    # Apply simple average blur
    filtered_image = pupil.filter_image_using_blur(
        image=image,
        kernel_size=7,
        border_type="default",
    )

    # Access results
    filtered_image_np = filtered_image.to_numpy()
    logger.success(
        "Applied Blur filter. Filtered output image shape: {}",
        filtered_image_np.shape,
    )

    # ===================== Visualization  (Optional) ======================

    # Initialize Rerun for visualization
    rr.init("filter_image_using_blur", spawn=True)
    rr.send_blueprint(
        rrb.Blueprint(
            rrb.Grid(
                rrb.Spatial2DView(name="Original", origin="input"),
                rrb.Spatial2DView(name="Filtered", origin="filtered_image"),
            ),
            rrb.SelectionPanel(),
            rrb.TimePanel(),
        ),
        make_active=True,
    )

    # Input image
    image_np = image.to_numpy()

    # Log images
    rr.log("input", rr.Image(image_np))
    rr.log("filtered_image", rr.Image(filtered_image_np))


# ===================== Enhancement =====================


# ===================== Contrast =====================


def enhance_image_using_clahe_example():
    """
    Applies Contrast Limited Adaptive Histogram Equalization.

    CLAHE enhances local contrast adaptively, preventing over-amplification
    of noise in uniform regions.

    The returned image is processed and used for visualization.
    """
    # ===================== Operation ==========================================

    # Load image
    filepath = str(DATA_DIR / "images" / "dark_warehouse.jpg")
    image = io.load_image(filepath=filepath)
    logger.success(f"Loaded image from {filepath}")

    # Enhance image using CLAHE
    filtered_image = pupil.enhance_image_using_clahe(
        image=image,
        clip_limit=10.0,
        tile_grid_size=8,
        color_space="lab",
    )

    # Access results
    filtered_image_np = filtered_image.to_numpy()
    logger.success(
        "Applied CLAHE filter. Enhanced output image shape: {}",
        filtered_image_np.shape,
    )

    # ===================== Visualization  (Optional) ======================

    # Initialize Rerun for visualization
    rr.init("enhance_image_using_clahe", spawn=True)
    rr.send_blueprint(
        rrb.Blueprint(
            rrb.Grid(
                rrb.Spatial2DView(name="Original", origin="input"),
                rrb.Spatial2DView(name="Enhanced", origin="enhanced_image"),
            ),
            rrb.SelectionPanel(),
            rrb.TimePanel(),
        ),
        make_active=True,
    )

    # Input image
    image_np = image.to_numpy()

    # Log images
    rr.log("input", rr.Image(image_np))
    rr.log("enhanced_image", rr.Image(filtered_image_np))


# ===================== Pyramid =====================


def transform_image_using_pyramid_downsampling_example():
    """
    Downsamples an image using Gaussian pyramid.

    Pyramid down reduces image resolution, useful for
    multi-scale analysis and efficient processing.

    The returned image is processed and used for visualization.
    """
    # ===================== Operation ==========================================

    # Load image
    filepath = str(DATA_DIR / "images" / "gearbox.png")
    image = io.load_image(filepath=filepath)
    logger.success(f"Loaded image from {filepath}")

    # Apply pyramid downsampling transformation multiple times to create a pyramid
    filtered_image = pupil.transform_image_using_pyramid_downsampling(
        image=image,
        scale_factor=0.5,
    )
    filtered_image_1 = pupil.transform_image_using_pyramid_downsampling(
        image=filtered_image,
        scale_factor=0.5,
    )
    filtered_image_2 = pupil.transform_image_using_pyramid_downsampling(
        image=filtered_image_1,
        scale_factor=0.5,
    )

    # Access results
    filtered_image_np = filtered_image.to_numpy()
    filtered_image_np_1 = filtered_image_1.to_numpy()
    filtered_image_np_2 = filtered_image_2.to_numpy()
    logger.success(
        "Applied pyramid downsampling. Transformed output image shapes: {}, {}, {}",
        filtered_image_np.shape,
        filtered_image_np_1.shape,
        filtered_image_np_2.shape,
    )

    # ===================== Visualization  (Optional) ======================

    # Initialize Rerun for visualization
    rr.init("transform_image_using_pyramid_downsampling", spawn=True)
    rr.send_blueprint(
        rrb.Blueprint(
            rrb.Grid(
                rrb.Spatial2DView(name="Original", origin="input"),
                rrb.Spatial2DView(name="Filtered 1", origin="filtered_image_1"),
                rrb.Spatial2DView(name="Filtered 2", origin="filtered_image_2"),
                rrb.Spatial2DView(name="Filtered 3", origin="filtered_image_3"),
            ),
            rrb.SelectionPanel(),
            rrb.TimePanel(),
        ),
        make_active=True,
    )

    # Input image
    image_np = image.to_numpy()

    # Log images
    rr.log("input", rr.Image(image_np))
    rr.log("filtered_image_1", rr.Image(filtered_image_np))
    rr.log("filtered_image_2", rr.Image(filtered_image_np_1))
    rr.log("filtered_image_3", rr.Image(filtered_image_np_2))


def transform_image_using_pyramid_upsampling_example():
    """
    Upsamples an image using Gaussian pyramid.

    Pyramid up increases image resolution, useful for image enlargement
    and multi-scale reconstruction.

    The returned image is processed and used for visualization.
    """
    # ===================== Operation ==========================================

    # Load image
    filepath = str(DATA_DIR / "images" / "buttons_arranged_downsampled.png")
    image = io.load_image(filepath=filepath)
    logger.success("Loaded image shape: {}", image.to_numpy().shape)

    # Apply pyramid upsampling transformation multiple times to create a pyramid
    filtered_image = pupil.transform_image_using_pyramid_upsampling(
        image=image,
        scale_factor=2.0,
    )
    filtered_image_1 = pupil.transform_image_using_pyramid_upsampling(
        image=filtered_image,
        scale_factor=2.0,
    )
    filtered_image_2 = pupil.transform_image_using_pyramid_upsampling(
        image=filtered_image_1,
        scale_factor=2.0,
    )

    # Access results
    filtered_image_np = filtered_image.to_numpy()
    filtered_image_np_1 = filtered_image_1.to_numpy()
    filtered_image_np_2 = filtered_image_2.to_numpy()
    logger.success(
        "Applied pyramid upsampling. Transformed output image shapes: {}, {}, {}",
        filtered_image_np.shape,
        filtered_image_np_1.shape,
        filtered_image_np_2.shape,
    )

    # ===================== Visualization  (Optional) ======================

    # Initialize Rerun for visualization
    rr.init("transform_image_using_pyramid_upsampling", spawn=True)
    rr.send_blueprint(
        rrb.Blueprint(
            rrb.Grid(
                rrb.Spatial2DView(name="Original", origin="input"),
                rrb.Spatial2DView(name="Filtered 1", origin="filtered_image_1"),
                rrb.Spatial2DView(name="Filtered 2", origin="filtered_image_2"),
                rrb.Spatial2DView(name="Filtered 3", origin="filtered_image_3"),
            ),
            rrb.SelectionPanel(),
            rrb.TimePanel(),
        ),
        make_active=True,
    )

    # Input image
    image_np = image.to_numpy()

    # Log images
    rr.log("input", rr.Image(image_np))
    rr.log("filtered_image_1", rr.Image(filtered_image_np))
    rr.log("filtered_image_2", rr.Image(filtered_image_np_1))
    rr.log("filtered_image_3", rr.Image(filtered_image_np_2))


# ===================== Thinning =====================


def transform_mask_using_blob_thinning_example():
    """
    Skeletonizes (thins) foreground blobs in a binary mask.

    This operation is useful for shape analysis, centerline extraction,
    and structural feature detection in binary masks.

    The returned image is processed and used for visualization.
    """
    # ===================== Operation ==========================================

    # Load image
    filepath = str(DATA_DIR / "images" / "handwriting_mask.png")
    image = io.load_image(filepath=filepath, as_binary=True)
    logger.success(f"Loaded image from {filepath}")

    # Apply Zhang-Suen thinning algorithm for skeletonization
    filtered_image = pupil.transform_mask_using_blob_thinning(
        image=image,
        thinning_type="thinning guohall",
    )

    # Access results
    filtered_image_np = filtered_image.to_numpy()
    logger.success(
        "Applied Zhang-Suen thinning. Transformed output image shape 1: {}",
        filtered_image_np.shape,
    )

    # ===================== Visualization  (Optional) ======================

    # Initialize Rerun for visualization
    rr.init("transform_mask_using_blob_thinning", spawn=True)
    rr.send_blueprint(
        rrb.Blueprint(
            rrb.Grid(
                rrb.Spatial2DView(name="Original", origin="input"),
                rrb.Spatial2DView(name="Filtered", origin="filtered_image"),
            ),
            rrb.SelectionPanel(),
            rrb.TimePanel(),
        ),
        make_active=True,
    )

    # Input image
    image_np = image.to_numpy()

    # Log images
    rr.log("input", rr.Image(image_np))
    rr.log("filtered_image", rr.Image(filtered_image_np))

    # ===================== Operation ==========================================

    # Load image
    filepath = str(DATA_DIR / "images" / "male_female_mask.png")
    image = io.load_image(filepath=filepath, as_binary=True)
    logger.success(f"Loaded image from {filepath}")

    # Apply Zhang-Suen thinning algorithm for skeletonization
    filtered_image = pupil.transform_mask_using_blob_thinning(
        image=image,
        thinning_type="thinning zhangsuen",
    )

    # Access results
    filtered_image_np = filtered_image.to_numpy()
    logger.success(
        "Applied Zhang-Suen thinning. Transformed output image shape 2: {}",
        filtered_image_np.shape,
    )

    # ===================== Visualization  (Optional) ======================

    # Initialize Rerun for visualization
    rr.init("transform_mask_using_blob_thinning_2", spawn=True)
    rr.send_blueprint(
        rrb.Blueprint(
            rrb.Grid(
                rrb.Spatial2DView(name="Original", origin="input"),
                rrb.Spatial2DView(name="Filtered", origin="filtered_image"),
            ),
            rrb.SelectionPanel(),
            rrb.TimePanel(),
        ),
        make_active=True,
    )

    # Input image
    image_np = image.to_numpy()

    # Log images
    rr.log("input", rr.Image(image_np))
    rr.log("filtered_image", rr.Image(filtered_image_np))


# ===================== Geomtery =====================


def calculate_image_pca_example():
    """
    Compute PCA on a binary mask and visualize centroid and principal axis
    exactly like the OpenCV implementation.
    """

    filepath = str(DATA_DIR / "images" / "can_vertical_6_mask.png")
    image = io.load_image(
        filepath=filepath, as_binary=True, binary_method="fixed"
    )
    logger.success(f"Loaded image from {filepath}")

    binary_mask = image.to_numpy()

    centroid, eigenvectors, eigenvalues, angle = pupil.calculate_image_pca(
        image=image
    )
    centroid = centroid.to_numpy()
    eigenvectors = eigenvectors.to_numpy()
    eigenvalues = eigenvalues.to_numpy()
    angle = angle.value

    rr.init("calculate_image_pca", spawn=True)

    rr.send_blueprint(
        rrb.Blueprint(
            rrb.Horizontal(
                rrb.Spatial2DView(name="Input", origin="input"),
                rrb.Spatial2DView(name="Output", origin="output"),
            )
        )
    )

    rr.log("input", rr.Image(binary_mask))
    rr.log("output/image", rr.Image(binary_mask))

    if (
        centroid is not None
        and eigenvectors is not None
        and eigenvalues is not None
    ):
        cx, cy = int(centroid[0]), int(centroid[1])

        # draw centroid
        rr.log(
            "output/centroid",
            rr.Points2D(
                positions=[[cx, cy]],
                colors=[[0, 255, 0]],
                radii=[8],
            ),
        )

        # same scale logic as OpenCV
        scale = (
            float(np.sqrt(np.real(eigenvalues[0])))
            if np.any(eigenvalues)
            else 30.0
        )

        ex = float(np.real(eigenvectors[0, 0])) * scale
        ey = float(np.real(eigenvectors[1, 0])) * scale

        pt2 = [cx + ex, cy + ey]

        # draw PCA line
        rr.log(
            "output/principal_axis",
            rr.LineStrips2D(
                [[[cx, cy], pt2]],
                colors=[[255, 0, 0]],
                radii=[2],
            ),
        )


def calculate_image_centroid_example():
    """
    Computes the centroid of non-zero pixels in a binary mask.
    """
    # ===================== Operation ==========================================

    filepath = str(DATA_DIR / "images" / "metal_part_mask.png")
    mask = io.load_image(
        filepath=filepath, as_binary=True, binary_method="fixed"
    )
    logger.success(f"Loaded mask from {filepath}")

    centroid = pupil.calculate_image_centroid(mask=mask)

    centroid_pos = centroid.to_numpy().reshape(-1, 2)
    logger.success(
        "Computed centroid. Position: ({}, {})",
        centroid_pos[0, 0],
        centroid_pos[0, 1],
    )

    # ===================== Visualization  (Optional) ======================

    rr.init("calculate_image_centroid", spawn=True)
    rr.send_blueprint(
        rrb.Blueprint(
            rrb.Grid(
                rrb.Spatial2DView(name="Mask", origin="mask"),
                rrb.Spatial2DView(name="Centroid", origin="centroid"),
            ),
            rrb.SelectionPanel(),
            rrb.TimePanel(),
        ),
        make_active=True,
    )

    mask_np = mask.to_numpy()
    rr.log("mask", rr.Image(mask_np))
    rr.log("centroid", rr.Image(mask_np))
    rr.log(
        "centroid/point",
        rr.Points2D(positions=centroid_pos, radii=4, colors=[[0, 255, 0]]),
    )


# ===================== Color =====================


def enhance_image_using_auto_gamma_correction_example():
    """
    Applies gamma correction to adjust image brightness non-linearly.

    This operation enhances details in dark or overexposed regions while
    preserving natural intensity relationships.

    The returned image is processed and used for visualization.
    """
    # ===================== Operation ==========================================

    # Load image
    filepath = str(DATA_DIR / "images" / "screws_in_dark_lighting.jpg")
    image = io.load_image(filepath=filepath)
    logger.success(f"Loaded image from {filepath}")

    # Apply auto gamma correction for non-linear brightness adjustment
    filtered_image = pupil.enhance_image_using_auto_gamma_correction(
        image=image,
    )

    # Access results
    filtered_image_np = filtered_image.to_numpy()
    logger.success(
        "Applied Auto Gamma Correction. Enhanced output image shape: {}",
        filtered_image_np.shape,
    )

    # ===================== Visualization  (Optional) ======================

    # Initialize Rerun for visualization
    rr.init("enhance_image_using_auto_gamma_correction", spawn=True)
    rr.send_blueprint(
        rrb.Blueprint(
            rrb.Grid(
                rrb.Spatial2DView(name="Original", origin="input"),
                rrb.Spatial2DView(name="Enhanced", origin="enhanced_image"),
            ),
            rrb.SelectionPanel(),
            rrb.TimePanel(),
        ),
        make_active=True,
    )

    # Input image
    image_np = image.to_numpy()

    # Log images
    rr.log("input", rr.Image(image_np))
    rr.log("enhanced_image", rr.Image(filtered_image_np))


def enhance_image_using_white_balance_example():
    """
    White balance (simple per-channel scaling).

    White balance adjusts the intensity of each color channel independently
    to reduce color casts and achieve a more neutral appearance. It is useful
    for correcting illumination bias and improving color consistency across images.

    The returned image is processed and used for visualization.
    """
    # ===================== Operation ==========================================

    # Load image
    filepath = str(DATA_DIR / "images" / "hand_tools_yellow_light.png")
    image = io.load_image(filepath=filepath)
    logger.success(f"Loaded image from {filepath}")

    # Apply white balance for color correction
    filtered_image = pupil.enhance_image_using_white_balance(
        image=image,
    )

    # Access results
    filtered_image_np = filtered_image.to_numpy()
    logger.success(
        "Applied White Balance. Enhanced output image shape: {}",
        filtered_image_np.shape,
    )

    # ===================== Visualization  (Optional) ======================

    # Initialize Rerun for visualization
    rr.init("enhance_image_using_white_balance", spawn=True)
    rr.send_blueprint(
        rrb.Blueprint(
            rrb.Grid(
                rrb.Spatial2DView(name="Original", origin="input"),
                rrb.Spatial2DView(name="Enhanced", origin="enhanced_image"),
            ),
            rrb.SelectionPanel(),
            rrb.TimePanel(),
        ),
        make_active=True,
    )

    # Input image
    image_np = image.to_numpy()

    # Log images
    rr.log("input", rr.Image(image_np))
    rr.log("enhanced_image", rr.Image(filtered_image_np))


def convert_image_color_space_example():
    """
    Converts an image between color spaces (e.g., BGR to RGB, RGB to HSV).
    """
    # ===================== Operation ==========================================

    filepath = str(DATA_DIR / "images" / "apples_black_container.jpg")
    image = io.load_image(filepath=filepath)
    logger.success(f"Loaded image from {filepath}")

    converted_image = pupil.convert_image_color_space(
        image=image,
        source_color_space="RGB",
        target_color_space="GRAY",
    )

    converted_image_np = converted_image.to_numpy()
    logger.success(
        "Converted color space. Output image shape: {}",
        converted_image_np.shape,
    )

    # ===================== Visualization  (Optional) ======================

    rr.init("convert_image_color_space", spawn=True)
    rr.send_blueprint(
        rrb.Blueprint(
            rrb.Grid(
                rrb.Spatial2DView(name="Original", origin="input"),
                rrb.Spatial2DView(name="Converted", origin="converted_image"),
            ),
            rrb.SelectionPanel(),
            rrb.TimePanel(),
        ),
        make_active=True,
    )

    image_np = image.to_numpy()
    rr.log("input", rr.Image(image_np))
    rr.log("converted_image", rr.Image(converted_image_np))


def normalize_image_intensity_example():
    """
    Normalizes image intensity values using minmax or histogram methods.
    """
    # ===================== Operation ==========================================

    filepath = str(DATA_DIR / "images" / "gauge_washed.png")
    image = io.load_image(filepath=filepath)
    logger.success(f"Loaded image from {filepath}")

    normalized_image = pupil.normalize_image_intensity(
        image=image,
        alpha=0.0,
        beta=255.0,
        normalization_method="minmax",
        output_format="8bit",
    )

    normalized_image_np = normalized_image.to_numpy()
    logger.success(
        "Normalized intensity. Output image shape: {}",
        normalized_image_np.shape,
    )

    # ===================== Visualization  (Optional) ======================

    rr.init("normalize_image_intensity", spawn=True)
    rr.send_blueprint(
        rrb.Blueprint(
            rrb.Grid(
                rrb.Spatial2DView(name="Original", origin="input"),
                rrb.Spatial2DView(name="Normalized", origin="normalized_image"),
            ),
            rrb.SelectionPanel(),
            rrb.TimePanel(),
        ),
        make_active=True,
    )

    image_np = image.to_numpy()
    rr.log("input", rr.Image(image_np))
    rr.log("normalized_image", rr.Image(normalized_image_np))


def split_image_into_channels_example():
    """
    Splits an image into its color channels, by default in the same order as the input image.
    For RGB images, the order is (R, G, B).
    For RGBA images, the order is (R, G, B, A).
    For BGR images, the order is (B, G, R).
    """
    # ===================== Operation ==========================================

    filepath = str(DATA_DIR / "images" / "vegetables.jpg")

    # Image loaded as RGB or RGBA always
    image = io.load_image(filepath=filepath)
    logger.success(f"Loaded image from {filepath}")

    image_channels = pupil.split_image_into_channels(image=image)

    # Convert ListOfImages → list[Image]
    channel_images = image_channels.to_list()

    # Convert to numpy
    channel_np_list = [img.to_numpy() for img in channel_images]

    logger.success(
        "Split channels. Number of channels: {}",
        len(channel_np_list),
    )

    # ===================== Visualization  (Optional) ======================

    rr.init("split_image_into_channels", spawn=True)

    # Create Spatial2DViews dynamically
    views = [rrb.Spatial2DView(name="Original", origin="input")]
    # Order of channels for this example is (R, G, B) as it loaded through the io.load_image function.
    channel_names = ["Red", "Green", "Blue"]
    for i in range(len(channel_np_list)):
        views.append(
            rrb.Spatial2DView(
                name=channel_names[i],
                origin=f"channel_{i + 1}",
            )
        )

    rr.send_blueprint(
        rrb.Blueprint(
            rrb.Grid(*views),
            rrb.SelectionPanel(),
            rrb.TimePanel(),
        ),
        make_active=True,
    )

    # Log original image
    image_np = image.to_numpy()
    rr.log("input", rr.Image(image_np))

    # Log each channel
    for i, channel_np in enumerate(channel_np_list):
        rr.log(f"channel_{i + 1}", rr.Image(channel_np))


def merge_image_from_channels_example():
    """
    Splits an image into channels, visualizes the channels,
    and then merges them back into a multi-channel image.
    """

    # ===================== Operation ==========================================

    filepath = str(DATA_DIR / "images" / "fruits_carts.jpg")
    image = io.load_image(filepath=filepath)
    logger.success(f"Loaded image from {filepath}")

    # Split image into channels
    image_channels = pupil.split_image_into_channels(image=image)

    # Convert ListOfImages → list[Image]
    channel_images = image_channels.to_list()

    # Convert channels to numpy
    channel_np_list = [img.to_numpy() for img in channel_images]

    logger.success(
        "Split channels. Number of channels: {}",
        len(channel_np_list),
    )

    # Merge channels back into an image
    merged_image = pupil.merge_image_from_channels(channels=channel_images)

    merged_image_np = merged_image.to_numpy()

    logger.success(
        "Merged {} channels. Output image shape: {}",
        len(channel_images),
        merged_image_np.shape,
    )

    # ===================== Visualization (Optional) ===========================

    rr.init("merge_image_channels", spawn=True)

    # Create Spatial2DViews dynamically
    views = []

    channel_names = ["Red", "Green", "Blue"]

    for i in range(len(channel_np_list)):
        views.append(
            rrb.Spatial2DView(
                name=channel_names[i],
                origin=f"channel_{i + 1}",
            )
        )

    views.append(
        rrb.Spatial2DView(
            name="Merged",
            origin="merged_image",
        )
    )

    rr.send_blueprint(
        rrb.Blueprint(
            rrb.Grid(*views),
            rrb.SelectionPanel(),
            rrb.TimePanel(),
        ),
        make_active=True,
    )

    # Log channel images
    for i, channel_np in enumerate(channel_np_list):
        rr.log(f"channel_{i + 1}", rr.Image(channel_np))

    # Log merged image
    rr.log("merged_image", rr.Image(merged_image_np))


# ===================== Transform =====================


def resize_image_example():
    """
    Resizes an image by a scale factor.
    """
    # Example 1: Resize image by scale factor
    # ===================== Operation ==========================================

    filepath = str(DATA_DIR / "images" / "gearbox.png")
    image = io.load_image(filepath=filepath)
    logger.success(f"Loaded image from {filepath}")

    resized_image = pupil.resize_image(
        image=image,
        scale_factor=0.5,
        interpolation_method="linear",
    )

    resized_image_np = resized_image.to_numpy()

    logger.success(
        "Resized image from input shape: {}, to output shape: {}",
        image.to_numpy().shape,
        resized_image_np.shape,
    )

    # ===================== Visualization  (Optional) ======================

    rr.init("resize_image_1", spawn=True)
    rr.send_blueprint(
        rrb.Blueprint(
            rrb.Grid(
                rrb.Spatial2DView(name="Original", origin="input"),
                rrb.Spatial2DView(name="Resized", origin="resized_image"),
            ),
            rrb.SelectionPanel(),
            rrb.TimePanel(),
        ),
        make_active=True,
    )

    image_np = image.to_numpy()
    rr.log("input", rr.Image(image_np))
    rr.log("resized_image", rr.Image(resized_image_np))

    # Example 2: Resize image by width and height
    # ===================== Operation ==========================================

    filepath = str(DATA_DIR / "images" / "gearbox.png")
    image = io.load_image(filepath=filepath)
    logger.success(f"Loaded image from {filepath}")

    resized_image = pupil.resize_image(
        image=image,
        resize_width=200,
        resize_height=200,
        interpolation_method="linear",
    )

    resized_image_np = resized_image.to_numpy()

    logger.success(
        "Resized image from input shape: {}, to output shape: {}",
        image.to_numpy().shape,
        resized_image_np.shape,
    )

    # ===================== Visualization  (Optional) ======================

    rr.init("resize_image_2", spawn=True)
    rr.send_blueprint(
        rrb.Blueprint(
            rrb.Grid(
                rrb.Spatial2DView(name="Original", origin="input"),
                rrb.Spatial2DView(name="Resized", origin="resized_image"),
            ),
            rrb.SelectionPanel(),
            rrb.TimePanel(),
        ),
        make_active=True,
    )

    image_np = image.to_numpy()
    rr.log("input", rr.Image(image_np))
    rr.log("resized_image", rr.Image(resized_image_np))


def resize_image_with_aspect_fit_example():
    """
    Resizes an image to fit within target dimensions while preserving aspect ratio.
    """
    # ===================== Operation ==========================================

    filepath = str(DATA_DIR / "images" / "gearbox.png")
    image = io.load_image(filepath=filepath)
    logger.success(f"Loaded image from {filepath}")

    resized_image = pupil.resize_image_with_aspect_fit(
        image=image,
        resize_width=400,
        resize_height=300,
        interpolation_method="linear",
    )

    resized_image_np = resized_image.to_numpy()
    logger.success(
        "Resized image (aspect fit) from input shape: {}, to output shape: {}",
        image.to_numpy().shape,
        resized_image_np.shape,
    )

    # ===================== Visualization  (Optional) ======================

    rr.init("resize_image_with_aspect_fit", spawn=True)
    rr.send_blueprint(
        rrb.Blueprint(
            rrb.Grid(
                rrb.Spatial2DView(name="Original", origin="input"),
                rrb.Spatial2DView(
                    name="Resized (Aspect Fit)", origin="resized_image"
                ),
            ),
            rrb.SelectionPanel(),
            rrb.TimePanel(),
        ),
        make_active=True,
    )

    image_np = image.to_numpy()
    rr.log("input", rr.Image(image_np))
    rr.log("resized_image", rr.Image(resized_image_np))


def rotate_image_example():
    """
    Rotates an image by an angle in degrees.
    """
    # ===================== Operation ==========================================

    filepath = str(DATA_DIR / "images" / "synthetic_data_bin.jpg")
    image = io.load_image(filepath=filepath)
    logger.success(f"Loaded image from {filepath}")

    rotated_image = pupil.rotate_image(
        image=image,
        angle_in_deg=10,
        interpolation_method="linear",
        keep_image_size=True,
    )

    rotated_image_np = rotated_image.to_numpy()
    logger.success("Rotated image. Output shape: {}", rotated_image_np.shape)

    # ===================== Visualization  (Optional) ======================

    rr.init("rotate_image", spawn=True)
    rr.send_blueprint(
        rrb.Blueprint(
            rrb.Grid(
                rrb.Spatial2DView(name="Original", origin="input"),
                rrb.Spatial2DView(name="Rotated", origin="rotated_image"),
            ),
            rrb.SelectionPanel(),
            rrb.TimePanel(),
        ),
        make_active=True,
    )

    image_np = image.to_numpy()
    rr.log("input", rr.Image(image_np))
    rr.log("rotated_image", rr.Image(rotated_image_np))


def translate_image_example():
    """
    Translates (shifts) an image by dx and dy pixels.
    """
    filepath = str(DATA_DIR / "images" / "checkerboard.jpg")
    image = io.load_image(filepath=filepath)
    logger.success(f"Loaded image from {filepath}")

    translated_image = pupil.translate_image(
        image=image,
        dx=100,
        dy=50,
        border_type="constant",
        border_value=0,
        interpolation_method="linear",
    )

    translated_image_np = translated_image.to_numpy()
    logger.success(
        "Translated image. Output shape: {}", translated_image_np.shape
    )

    rr.init("translate_image", spawn=True)
    rr.send_blueprint(
        rrb.Blueprint(
            rrb.Grid(
                rrb.Spatial2DView(name="Original", origin="input"),
                rrb.Spatial2DView(name="Translated", origin="translated_image"),
            ),
            rrb.SelectionPanel(),
            rrb.TimePanel(),
        ),
        make_active=True,
    )
    rr.log("input", rr.Image(image.to_numpy()))
    rr.log("translated_image", rr.Image(translated_image_np))


def pad_image_example():
    """
    Pads an image on top, bottom, left, and right.
    """
    filepath = str(DATA_DIR / "images" / "bin_picking_metal_2.jpg")
    image = io.load_image(filepath=filepath)
    logger.success(f"Loaded image from {filepath}")

    padded_image = pupil.pad_image(
        image=image,
        top=200,
        bottom=50,
        left=100,
        right=75,
        border_type="constant",
        border_value=0.0,
    )

    padded_image_np = padded_image.to_numpy()
    logger.success("Padded image. Output shape: {}", padded_image_np.shape)

    rr.init("pad_image", spawn=True)
    rr.send_blueprint(
        rrb.Blueprint(
            rrb.Grid(
                rrb.Spatial2DView(name="Original", origin="input"),
                rrb.Spatial2DView(name="Padded", origin="padded_image"),
            ),
            rrb.SelectionPanel(),
            rrb.TimePanel(),
        ),
        make_active=True,
    )
    rr.log("input", rr.Image(image.to_numpy()))
    rr.log("padded_image", rr.Image(padded_image_np))


def crop_image_center_example():
    """
    Center-crops an image to the specified dimensions.
    """
    filepath = str(DATA_DIR / "images" / "rusted_metal_gear.jpg")
    image = io.load_image(filepath=filepath)
    logger.success(f"Loaded image from {filepath}")

    cropped_image = pupil.crop_image_center(
        image=image,
        crop_width=300,
        crop_height=300,
        pad_color=(0, 0, 0),
    )

    cropped_image_np = cropped_image.to_numpy()
    logger.success(
        "Center-cropped image. Output shape: {}", cropped_image_np.shape
    )

    rr.init("crop_image_center", spawn=True)
    rr.send_blueprint(
        rrb.Blueprint(
            rrb.Grid(
                rrb.Spatial2DView(name="Original", origin="input"),
                rrb.Spatial2DView(
                    name="Center Cropped", origin="cropped_image"
                ),
            ),
            rrb.SelectionPanel(),
            rrb.TimePanel(),
        ),
        make_active=True,
    )
    rr.log("input", rr.Image(image.to_numpy()))
    rr.log("cropped_image", rr.Image(cropped_image_np))


def crop_image_using_bounding_boxes_example():
    """
    Crops an image using multiple bounding boxes.
    """

    # ===================== Operation ==========================================

    filepath = str(DATA_DIR / "images" / "driver_screw.png")
    image = io.load_image(filepath=filepath)
    logger.success(f"Loaded image from {filepath}")

    image_np = image.to_numpy()
    h, w = image_np.shape[:2]

    bounding_boxes = [
        [65, 235, 330, 240],
        [370, 35, 330, 155],
        [445, 210, 85, 300],
    ]

    cropped_images = pupil.crop_image_using_bounding_boxes(
        image=image,
        bounding_boxes=bounding_boxes,
        retain_coordinates=True,
    )

    num_crops = len(cropped_images.to_list())
    logger.success("Cropped {} regions", num_crops)

    # ===================== Visualization (Optional) ======================

    rr.init("crop_image_boxes", spawn=True)

    views = [rrb.Spatial2DView(name="Original", origin="input")]

    for i in range(num_crops):
        views.append(
            rrb.Spatial2DView(name=f"Crop {i + 1}", origin=f"crops/{i}")
        )

    rr.send_blueprint(
        rrb.Blueprint(
            rrb.Horizontal(*views),
            rrb.SelectionPanel(),
            rrb.TimePanel(),
        ),
        make_active=True,
    )

    # Log original image
    rr.log("input", rr.Image(image_np))

    # Log cropped images
    for i, crop in enumerate(cropped_images.to_list()):
        rr.log(f"crops/{i}", rr.Image(crop.to_numpy()))


def crop_image_using_polygon_example():
    """
    Crops an image using a polygon mask.
    """
    # ===================== Operation ==========================================

    filepath = str(DATA_DIR / "images" / "pedestrians.jpg")
    image = io.load_image(filepath=filepath)
    logger.success(f"Loaded image from {filepath}")

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

    cropped_image = pupil.crop_image_using_polygon(
        image=image,
        polygon_vertices=polygon_vertices,
    )

    cropped_image_np = cropped_image.to_numpy()
    logger.success(
        "Cropped image (polygon). Output shape: {}", cropped_image_np.shape
    )

    # ===================== Visualization  (Optional) ======================

    rr.init("crop_image_using_polygon", spawn=True)
    rr.send_blueprint(
        rrb.Blueprint(
            rrb.Grid(
                rrb.Spatial2DView(name="Original", origin="input"),
                rrb.Spatial2DView(name="Cropped", origin="cropped_image"),
            ),
            rrb.SelectionPanel(),
            rrb.TimePanel(),
        ),
        make_active=True,
    )

    image_np = image.to_numpy()
    rr.log("input", rr.Image(image_np))
    rr.log("cropped_image", rr.Image(cropped_image_np))


# ===================== Operations (Bitwise / Overlay) =====================


def bitwise_and_images_example():
    """
    Performs bitwise AND between two images.
    """
    # ===================== Operation ==========================================

    filepath_1 = str(DATA_DIR / "images" / "bin_picking_metal_2.jpg")

    image_a = io.load_image(filepath=filepath_1)

    # Create empty mask with bounding box
    bbox = [450, 210, 1040, 616]
    x1, y1, x2, y2 = bbox
    mask = np.zeros(image_a.to_numpy().shape[:2], dtype=np.uint8)
    mask[y1:y2, x1:x2] = 255

    result = pupil.bitwise_and_images(image_a=image_a, image_b=mask)

    result_np = result.to_numpy()
    logger.success("Bitwise AND. Output shape: {}", result_np.shape)

    # ===================== Visualization  (Optional) ======================

    rr.init("bitwise_and_images", spawn=True)
    rr.send_blueprint(
        rrb.Blueprint(
            rrb.Horizontal(
                rrb.Spatial2DView(name="Image A", origin="input_a"),
                rrb.Spatial2DView(name="Image B", origin="input_b"),
                rrb.Spatial2DView(name="Bitwise AND", origin="result"),
            ),
            rrb.SelectionPanel(),
            rrb.TimePanel(),
        ),
        make_active=True,
    )

    image_a_np = image_a.to_numpy()
    image_b_np = mask
    rr.log("input_a", rr.Image(image_a_np))
    rr.log("input_b", rr.Image(image_b_np))
    rr.log("result", rr.Image(result.to_numpy()))


def bitwise_or_images_example():
    """
    Performs bitwise OR between two images.
    """
    # ===================== Operation ==========================================

    filepath_1 = str(DATA_DIR / "images" / "can_vertical_6_mask.png")
    filepath_2 = str(DATA_DIR / "images" / "rectangles_mask.png")

    image_a = io.load_image(
        filepath=filepath_1, as_binary=True, binary_method="fixed"
    )
    image_b = io.load_image(
        filepath=filepath_2, as_binary=True, binary_method="fixed"
    )

    image_b = pupil.resize_image_with_aspect_fit(
        image=image_b,
        resize_width=image_a.width,
        resize_height=image_a.height,
        pad_color=(0, 0, 0),
    )
    result = pupil.bitwise_or_images(image_a=image_a, image_b=image_b)

    result_np = result.to_numpy()
    logger.success("Bitwise OR. Output shape: {}", result_np.shape)

    # ===================== Visualization  (Optional) ======================

    rr.init("bitwise_or_images", spawn=True)
    rr.send_blueprint(
        rrb.Blueprint(
            rrb.Horizontal(
                rrb.Spatial2DView(name="Image A", origin="input_a"),
                rrb.Spatial2DView(name="Image B", origin="input_b"),
                rrb.Spatial2DView(name="Bitwise OR", origin="result"),
            ),
            rrb.SelectionPanel(),
            rrb.TimePanel(),
        ),
        make_active=True,
    )

    image_a_np = image_a.to_numpy()
    image_b_np = image_b.to_numpy()
    rr.log("input_a", rr.Image(image_a_np))
    rr.log("input_b", rr.Image(image_b_np))
    rr.log("result", rr.Image(result_np))


def bitwise_xor_images_example():
    """
    Performs bitwise XOR between two images.
    """
    # ===================== Operation ==========================================

    filepath_1 = str(DATA_DIR / "images" / "image_1.png")
    filepath_2 = str(DATA_DIR / "images" / "image_2.png")
    image_a = io.load_image(
        filepath=filepath_1, as_binary=True, binary_method="fixed"
    )
    image_b = io.load_image(
        filepath=filepath_2, as_binary=True, binary_method="fixed"
    )

    image_2_resized = pupil.resize_image_with_aspect_fit(
        image=image_b,
        resize_width=image_a.width,
        resize_height=image_a.height,
    )
    result = pupil.bitwise_xor_images(image_a=image_a, image_b=image_2_resized)

    result_np = result.to_numpy()
    logger.success("Bitwise XOR. Output shape: {}", result_np.shape)

    # ===================== Visualization  (Optional) ======================

    rr.init("bitwise_xor_images", spawn=True)
    rr.send_blueprint(
        rrb.Blueprint(
            rrb.Horizontal(
                rrb.Spatial2DView(name="Image A", origin="input_a"),
                rrb.Spatial2DView(name="Image B", origin="input_b"),
                rrb.Spatial2DView(name="Bitwise XOR", origin="result"),
            ),
            rrb.SelectionPanel(),
            rrb.TimePanel(),
        ),
        make_active=True,
    )

    image_a_np = image_a.to_numpy()
    image_b_np = image_b.to_numpy()
    rr.log("input_a", rr.Image(image_a_np))
    rr.log("input_b", rr.Image(image_b_np))
    rr.log("result", rr.Image(result_np))


def bitwise_difference_images_example():
    """
    Performs absolute difference between two images.
    """
    # ===================== Operation ==========================================

    filepath_1 = str(DATA_DIR / "images" / "driver_screw.png")
    filepath_2 = str(DATA_DIR / "images" / "difference_image.png")
    image_a = io.load_image(
        filepath=filepath_1, as_binary=True, binary_method="otsu"
    )
    image_b = io.load_image(
        filepath=filepath_2, as_binary=True, binary_method="otsu"
    )

    image_b_resized = pupil.resize_image_with_aspect_fit(
        image=image_b,
        resize_width=image_a.width,
        resize_height=image_a.height,
    )
    result = pupil.bitwise_difference_images(
        image_a=image_a, image_b=image_b_resized
    )

    result_np = result.to_numpy()
    logger.success("Bitwise difference. Output shape: {}", result_np.shape)

    # ===================== Visualization  (Optional) ======================

    rr.init("bitwise_difference_images", spawn=True)
    rr.send_blueprint(
        rrb.Blueprint(
            rrb.Horizontal(
                rrb.Spatial2DView(name="Image A", origin="input_a"),
                rrb.Spatial2DView(name="Image B", origin="input_b"),
                rrb.Spatial2DView(name="Bitwise Difference", origin="result"),
            ),
            rrb.SelectionPanel(),
            rrb.TimePanel(),
        ),
        make_active=True,
    )

    image_a_np = image_a.to_numpy()
    image_b_np = image_b_resized.to_numpy()
    rr.log("input_a", rr.Image(image_a_np))
    rr.log("input_b", rr.Image(image_b_np))
    rr.log("result", rr.Image(result_np))


def bitwise_not_image_example():
    """
    Performs bitwise NOT (inversion) on an image.
    """
    # ===================== Operation ==========================================

    filepath = str(DATA_DIR / "images" / "einstein.png")
    image = io.load_image(
        filepath=filepath, as_binary=True, binary_method="fixed"
    )
    logger.success(f"Loaded image from {filepath}")

    result = pupil.bitwise_not_image(image=image)

    result_np = result.to_numpy()
    logger.success("Bitwise NOT. Output shape: {}", result_np.shape)

    # ===================== Visualization  (Optional) ======================

    rr.init("bitwise_not_image", spawn=True)
    rr.send_blueprint(
        rrb.Blueprint(
            rrb.Grid(
                rrb.Spatial2DView(name="Original", origin="input"),
                rrb.Spatial2DView(name="Inverted", origin="result"),
            ),
            rrb.SelectionPanel(),
            rrb.TimePanel(),
        ),
        make_active=True,
    )

    image_np = image.to_numpy()
    rr.log("input", rr.Image(image_np))
    rr.log("result", rr.Image(result_np))


def overlay_images_using_weighted_overlay_example():
    """
    Blends two images using weighted overlay.
    """
    # ===================== Operation ==========================================

    filepath = str(DATA_DIR / "images" / "rusted_metal_gear.jpg")
    image_a = io.load_image(
        filepath=filepath
    )
    image_a = pupil.resize_image_with_aspect_fit(
        image=image_a,
        resize_width=512,
        resize_height=512,
    )

    image_b = pupil.rotate_image(
        image=image_a, angle_in_deg=60.0, keep_image_size=True
    )
    logger.success(f"Loaded image from {filepath}")

    blended = pupil.overlay_images_using_weighted_overlay(
        image_a=image_a,
        image_b=image_b,
        weight_a=0.5,
        weight_b=0.5,
    )

    blended_np = blended.to_numpy()
    logger.success("Weighted overlay. Output shape: {}", blended_np.shape)

    # ===================== Visualization  (Optional) ======================

    rr.init("overlay_images_using_weighted_overlay", spawn=True)
    rr.send_blueprint(
        rrb.Blueprint(
            rrb.Grid(
                rrb.Spatial2DView(name="Image A", origin="input_a"),
                rrb.Spatial2DView(name="Image B", origin="input_b"),
                rrb.Spatial2DView(name="Blended", origin="blended"),
            ),
            rrb.SelectionPanel(),
            rrb.TimePanel(),
        ),
        make_active=True,
    )

    image_np = image_a.to_numpy()
    image_2_np = image_b.to_numpy()
    rr.log("input_a", rr.Image(image_np))
    rr.log("input_b", rr.Image(image_2_np))
    rr.log("blended", rr.Image(blended_np))


# ===================== Projection =====================


def project_pixel_to_camera_point_example():
    """
    Projects a pixel and depth to a 3D point in camera coordinates.
    """
    # ===================== Operation ==========================================

    camera_intrinsics = np.array(
        [[500.0, 0, 320.0], [0, 500.0, 240.0], [0, 0, 1.0]],
        dtype=np.float64,
    )
    distortion_coefficients = np.array(
        [0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64
    )
    pixel = np.array([320.0, 240.0], dtype=np.float64)
    depth = 1.0

    camera_T_point = pupil.project_pixel_to_camera_point(
        camera_intrinsics=camera_intrinsics,
        distortion_coefficients=distortion_coefficients,
        pixel=pixel,
        depth=depth,
    )

    logger.success(
        "Projected pixel to camera point. camera_T_point shape: {}",
        np.asarray(camera_T_point.matrix).shape
        if hasattr(camera_T_point, "matrix")
        else "N/A",
    )

    # ===================== Visualization  (Optional) ======================

    rr.init("project_pixel_to_camera_point", spawn=True)
    rr.send_blueprint(
        rrb.Blueprint(
            rrb.Grid(rrb.Spatial3DView(name="Camera Point", origin="result")),
            rrb.SelectionPanel(),
            rrb.TimePanel(),
        ),
        make_active=True,
    )

    matrix = (
        camera_T_point.matrix
        if hasattr(camera_T_point, "matrix")
        else np.asarray(camera_T_point)
    )
    point_3d = matrix[:3, 3].reshape(1, 3).astype(np.float32)
    rr.log("result", rr.Points3D(positions=point_3d, colors=(0, 255, 0)))


def project_camera_point_to_pixel_example():
    """
    Projects a 3D point in camera coordinates to pixel coordinates.
    """
    # ===================== Operation ==========================================

    camera_intrinsics = np.array(
        [[500.0, 0, 320.0], [0, 500.0, 240.0], [0, 0, 1.0]],
        dtype=np.float64,
    )
    distortion_coefficients = np.array(
        [0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64
    )
    point = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    pixel = pupil.project_camera_point_to_pixel(
        camera_intrinsics=camera_intrinsics,
        distortion_coefficients=distortion_coefficients,
        point=point,
    )

    positions = pixel.to_numpy().reshape(-1, 2)
    logger.success("Projected camera point to pixel. Pixel: {}", positions)

    # ===================== Visualization  (Optional) ======================

    rr.init("project_camera_point_to_pixel", spawn=True)
    rr.send_blueprint(
        rrb.Blueprint(
            rrb.Grid(rrb.Spatial2DView(name="Pixel", origin="pixel")),
            rrb.SelectionPanel(),
            rrb.TimePanel(),
        ),
        make_active=True,
    )

    canvas = np.ones((480, 640, 3), dtype=np.uint8)
    rr.log("pixel", rr.Image(canvas))
    rr.log(
        "pixel/projected_point",
        rr.Points2D(
            positions=positions,
            radii=6,
        ),
    )


def project_pixel_to_world_point_example():
    """
    Projects a pixel and depth to a 3D point in world coordinates.
    """
    # ===================== Operation ==========================================

    camera_intrinsics = np.array(
        [[500.0, 0, 320.0], [0, 500.0, 240.0], [0, 0, 1.0]],
        dtype=np.float64,
    )
    distortion_coefficients = np.array(
        [0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64
    )
    pixel = np.array([320.0, 240.0], dtype=np.float64)
    depth = 1.0
    world_T_camera = np.eye(4, dtype=np.float64)
    world_T_camera[2, 3] = 1.0

    world_T_point = pupil.project_pixel_to_world_point(
        camera_intrinsics=camera_intrinsics,
        distortion_coefficients=distortion_coefficients,
        pixel=pixel,
        depth=depth,
        world_T_camera=world_T_camera,
    )

    logger.success(
        "Projected pixel to world point. world_T_point shape: {}",
        np.asarray(world_T_point.matrix).shape
        if hasattr(world_T_point, "matrix")
        else "N/A",
    )

    # ===================== Visualization  (Optional) ======================

    rr.init("project_pixel_to_world_point", spawn=True)
    rr.send_blueprint(
        rrb.Blueprint(
            rrb.Grid(rrb.Spatial3DView(name="World Point", origin="result")),
            rrb.SelectionPanel(),
            rrb.TimePanel(),
        ),
        make_active=True,
    )

    matrix = (
        world_T_point.matrix
        if hasattr(world_T_point, "matrix")
        else np.asarray(world_T_point)
    )
    point_3d = matrix[:3, 3].reshape(1, 3).astype(np.float32)
    rr.log("result", rr.Points3D(positions=point_3d, colors=(0, 255, 0)))


def project_world_point_to_pixel_example():
    """
    Projects a 3D point in world coordinates to pixel coordinates.
    """
    # ===================== Operation ==========================================

    camera_intrinsics = np.array(
        [[500.0, 0, 320.0], [0, 500.0, 240.0], [0, 0, 1.0]],
        dtype=np.float64,
    )
    distortion_coefficients = np.array(
        [0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64
    )
    point = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    world_T_camera = np.eye(4, dtype=np.float64)
    world_T_camera[2, 3] = 1.0

    pixel = pupil.project_world_point_to_pixel(
        camera_intrinsics=camera_intrinsics,
        distortion_coefficients=distortion_coefficients,
        point=point,
        world_T_camera=world_T_camera,
    )

    positions = pixel.to_numpy().reshape(-1, 2)
    logger.success("Projected world point to pixel. Pixel: {}", positions)

    # ===================== Visualization  (Optional) ======================

    rr.init("project_world_point_to_pixel", spawn=True)
    rr.send_blueprint(
        rrb.Blueprint(
            rrb.Grid(rrb.Spatial2DView(name="Pixel", origin="pixel")),
            rrb.SelectionPanel(),
            rrb.TimePanel(),
        ),
        make_active=True,
    )
    canvas = np.ones((480, 640, 3), dtype=np.uint8)

    rr.log("pixel", rr.Image(canvas))

    rr.log(
        "pixel/projected_point",
        rr.Points2D(
            positions=positions,
            radii=6,
        ),
    )


def get_example_dict():
    """Returns a dictionary mapping example names (without _example suffix) to their functions."""
    return {
        # Filter Examples
        # Morphology Examples
        "filter_image_using_morphological_erode": filter_image_using_morphological_erode_example,
        "filter_image_using_morphological_dilate": filter_image_using_morphological_dilate_example,
        "filter_image_using_morphological_close": filter_image_using_morphological_close_example,
        "filter_image_using_morphological_open": filter_image_using_morphological_open_example,
        "filter_image_using_morphological_gradient": filter_image_using_morphological_gradient_example,
        "filter_image_using_morphological_tophat": filter_image_using_morphological_tophat_example,
        "filter_image_using_morphological_blackhat": filter_image_using_morphological_blackhat_example,
        "filter_image_using_morphological_hitmiss": filter_image_using_morphological_hitmiss_example,
        # Ridge Examples
        "filter_image_using_frangi": filter_image_using_frangi_example,
        "filter_image_using_hessian": filter_image_using_hessian_example,
        "filter_image_using_sato": filter_image_using_sato_example,
        "filter_image_using_meijering": filter_image_using_meijering_example,
        # Sharpening Examples
        "filter_image_using_laplacian": filter_image_using_laplacian_example,
        "filter_image_using_sobel": filter_image_using_sobel_example,
        "filter_image_using_scharr": filter_image_using_scharr_example,
        "filter_image_using_gabor": filter_image_using_gabor_example,
        # Smoothing Examples
        "filter_image_using_bilateral": filter_image_using_bilateral_example,
        "filter_image_using_box": filter_image_using_box_example,
        "filter_image_using_gaussian_blur": filter_image_using_gaussian_blur_example,
        "filter_image_using_median_blur": filter_image_using_median_blur_example,
        "filter_image_using_blur": filter_image_using_blur_example,
        # Enhancement Examples
        "enhance_image_using_auto_gamma_correction": enhance_image_using_auto_gamma_correction_example,
        "enhance_image_using_white_balance": enhance_image_using_white_balance_example,
        # Contrast Examples
        "enhance_image_using_clahe": enhance_image_using_clahe_example,
        # Pyramid Examples
        "transform_image_using_pyramid_downsampling": transform_image_using_pyramid_downsampling_example,
        "transform_image_using_pyramid_upsampling": transform_image_using_pyramid_upsampling_example,
        # Thinning Examples
        "transform_mask_using_blob_thinning": transform_mask_using_blob_thinning_example,
        # Geometry Examples
        "calculate_image_pca": calculate_image_pca_example,
        "calculate_image_centroid": calculate_image_centroid_example,
        # Color Examples
        "convert_image_color_space": convert_image_color_space_example,
        "normalize_image_intensity": normalize_image_intensity_example,
        "split_image_into_channels": split_image_into_channels_example,
        "merge_image_from_channels": merge_image_from_channels_example,
        # Transform Examples
        "resize_image": resize_image_example,
        "resize_image_with_aspect_fit": resize_image_with_aspect_fit_example,
        "rotate_image": rotate_image_example,
        "translate_image": translate_image_example,
        "pad_image": pad_image_example,
        "crop_image_center": crop_image_center_example,
        "crop_image_using_bounding_boxes": crop_image_using_bounding_boxes_example,
        "crop_image_using_polygon": crop_image_using_polygon_example,
        # Operations (Bitwise / Overlay) Examples
        "bitwise_and_images": bitwise_and_images_example,
        "bitwise_or_images": bitwise_or_images_example,
        "bitwise_xor_images": bitwise_xor_images_example,
        "bitwise_difference_images": bitwise_difference_images_example,
        "bitwise_not_image": bitwise_not_image_example,
        "overlay_images_using_weighted_overlay": overlay_images_using_weighted_overlay_example,
        # Projection Examples
        "project_pixel_to_camera_point": project_pixel_to_camera_point_example,
        "project_camera_point_to_pixel": project_camera_point_to_pixel_example,
        "project_pixel_to_world_point": project_pixel_to_world_point_example,
        "project_world_point_to_pixel": project_world_point_to_pixel_example,
    }


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run pupil examples")
    parser.add_argument(
        "--example",
        type=str,
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
            logger.info(f"  - {example_name}")
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
            logger.error("Did you mean one of these?")
            for match in close_matches:
                logger.error(f"  - {match}")

        raise SystemExit(1)

    logger.info(f"Running {args.example} example...")
    example_dict[args.example]()
    logger.success(f"Example {args.example} completed.")


if __name__ == "__main__":
    main()
