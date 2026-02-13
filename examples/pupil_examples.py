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
    filepath = str(DATA_DIR / "images" / "brain_scan.jpg")
    image = io.load_image(filepath=filepath)
    logger.success(f"Loaded image from {filepath}")
    
    # Enhance image using CLAHE
    filtered_image = pupil.enhance_image_using_clahe(
        image=image,
        clip_limit=10.0,
        tile_grid_size=4,
        color_space="lab",
    )

    # Access results
    filtered_image_np = filtered_image.to_numpy()
    logger.success("Applied CLAHE filter. Enhanced output image shape: {}", filtered_image_np.shape)
    
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
    image = io.load_image(filepath=filepath, as_binary=True, binary_method='fixed')
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
    logger.success("Applied erosion morphological operation. Output image shape: {}", filtered_image_np.shape)
    
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
    image = io.load_image(filepath=filepath, as_binary=True, binary_method='fixed')
    logger.success(f"Loaded image from {filepath}")
    
    # Apply dilation morphological operation
    filtered_image = pupil.filter_image_using_morphological_dilate(
        image=image,
        kernel_size=5,
        kernel_shape="ellipse",
        iterations=5,
        border_type="default",
    )
    
    # Access results
    filtered_image_np = filtered_image.to_numpy()
    logger.success("Applied dilation morphological operation. Output image shape: {}", filtered_image_np.shape)
    
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
    logger.success("Applied close morphological operation. Output image shape: {}", filtered_image_np.shape)
    
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
    image = io.load_image(filepath=filepath, as_binary=True, binary_method='fixed')
    logger.success(f"Loaded image from {filepath}")

    # Apply open morphological operation
    filtered_image = pupil.filter_image_using_morphological_open(
        image=image,
        kernel_size=3,
        kernel_shape="ellipse",
        iterations=2,
        border_type="default",
    )

    # Access results
    filtered_image_np = filtered_image.to_numpy()
    logger.success("Applied open morphological operation. Output image shape: {}", filtered_image_np.shape)
    
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
    logger.success("Applied gradient morphological operation. Output image shape: {}", filtered_image_np.shape)
    
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
    logger.success("Applied tophat morphological operation. Output image shape: {}", filtered_image_np.shape)
    
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
    logger.success("Applied blackhat morphological operation. Output image shape: {}", filtered_image_np.shape)
    
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
    logger.success("Applied pyramid downsampling. Transformed output image shapes: {}, {}, {}", 
                   filtered_image_np.shape, filtered_image_np_1.shape, filtered_image_np_2.shape)
    
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
    logger.success("Applied pyramid upsampling. Transformed output image shapes: {}, {}, {}", filtered_image_np.shape, filtered_image_np_1.shape, filtered_image_np_2.shape)
    
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
        constant_value=0.0,
    )

    # Access results
    filtered_image_np = filtered_image.to_numpy()
    logger.success("Applied Frangi filter. Filtered output image shape: {}", filtered_image_np.shape)

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
        detect_black_ridges=True,
        border_type="reflect",
        constant_value=0.0,
    )

    # Access results
    filtered_image_np = filtered_image.to_numpy()
    logger.success("Applied Hessian filter. Filtered output image shape: {}", filtered_image_np.shape)

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
        constant_value=0.0,
    )

    # Access results
    filtered_image_np = filtered_image.to_numpy()
    logger.success("Applied Sato filter. Filtered output image shape: {}", filtered_image_np.shape)

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
        constant_value=0.0,
    )

    # Access results
    filtered_image_np = filtered_image.to_numpy()
    logger.success("Applied Meijering filter. Filtered output image shape: {}", filtered_image_np.shape)

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
    filepath = str(DATA_DIR / "images" / "flat_mechanical_component_denoised.png")
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
    logger.debug("Laplacian filter output image dtype: {}", filtered_image_np.dtype)
    logger.success("Applied Laplacian filter. Filtered output image shape: {}", filtered_image_np.shape)
    
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
    logger.success("Applied Sobel filter. Filtered output image shape: {}", filtered_image_np.shape)
    
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
        output_format="8bit",
        dx=0,
        dy=1,
        scale=1.0,
        delta=0.0,
        border_type="default",
    )

    # Access results
    filtered_image_np = filtered_image.to_numpy()
    logger.success("Applied Scharr filter. Filtered output image shape: {}", filtered_image_np.shape)

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
        standard_deviation=15.0,
        orientation=0.0,
        wavelength=10.0,
        aspect_ratio=0.5,
        phase_offset=np.pi * 0.5,
    )

    # Access results
    filtered_image_np = filtered_image.to_numpy()
    logger.success("Applied Gabor filter. Filtered output image shape: {}", filtered_image_np.shape)
    
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
        neighborhood_diameter=19,
        spatial_sigma=75.0,
        color_intensity_sigma=100.0,
        border_type="default",
    )  

    # Access results
    filtered_image_np = filtered_image.to_numpy()
    logger.success("Applied Bilateral filter. Filtered output image shape: {}", filtered_image_np.shape)
    
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
    logger.success("Applied Median Blur filter. Filtered output image shape: {}", filtered_image_np.shape)
    
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
    logger.success("Applied Box filter. Filtered output image shape: {}", filtered_image_np.shape)
    
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
    logger.success("Applied Gaussian Blur filter. Filtered output image shape: {}", filtered_image_np.shape)
    
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
    logger.success("Applied Blur filter. Filtered output image shape: {}", filtered_image_np.shape)
    
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
    logger.success("Applied Auto Gamma Correction. Enhanced output image shape: {}", filtered_image_np.shape)
    
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
    logger.success("Applied White Balance. Enhanced output image shape: {}", filtered_image_np.shape)
    
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
        thinning_type="thinning_zhangsuen",
    )

    # Access results
    filtered_image_np = filtered_image.to_numpy()
    logger.success("Applied Zhang-Suen thinning. Transformed output image shape 1: {}", filtered_image_np.shape)
    
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
        thinning_type="thinning_zhangsuen",
    )

    # Access results
    filtered_image_np = filtered_image.to_numpy()
    logger.success("Applied Zhang-Suen thinning. Transformed output image shape 2: {}", filtered_image_np.shape)
    
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
            logger.error(f"Did you mean one of these?")
            for match in close_matches:
                logger.error(f"  - {match}")

        raise SystemExit(1)

    logger.info(f"Running {args.example} example...")
    example_dict[args.example]()
    logger.success(f"Example {args.example} completed.")


if __name__ == "__main__":
    main()

