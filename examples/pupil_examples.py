import argparse
import difflib
import pathlib
import numpy as np

from loguru import logger
import rerun as rr

from telekinesis import pupil
from datatypes import datatypes, io

ROOT_DIR = pathlib.Path(__file__).parent.parent.resolve()
DATA_DIR = ROOT_DIR / "telekinesis-data"

# Helper function for rerun initialization
def init_rerun(example_name: str):
    """
    Initialize Rerun for visualization.
    
    Attempts to connect to an existing Rerun viewer instance. If no instance
    is available, spawns a new Rerun viewer window. This function should be
    called before logging any visualization data to Rerun.
    
    Args:
        example_name: A unique name for this visualization session. Used to
            identify the recording in Rerun.
    """
    rr.init(example_name, spawn=False)
    try:
        rr.connect()
    except Exception:
        rr.spawn()


# Helper functions for loading data
def load_image(filename, as_gray=False, as_binary=False, binary_method='otsu') -> datatypes.Image:
    """
    Load an image from the default data directory.
    
    Loads an image file from the telekinesis-data/images directory. Supports
    loading images in different formats (color, grayscale, binary) and various
    binary thresholding methods.
    
    Args:
        filename: Name of the image file to load (e.g., "brain_scan.jpg").
        as_gray: If True, convert the image to grayscale. Default: False.
        as_binary: If True, convert the image to binary (black and white).
            Default: False.
        binary_method: Method for binary thresholding when as_binary=True.
            Options: 'otsu', 'fixed', etc. Default: 'otsu'.
    
    Returns:
        A datatypes.Image object containing the loaded image, or None if the
        file is not found or cannot be loaded.
    
    Note:
        The image is loaded from DATA_DIR / "images" / filename. If the file
        doesn't exist, a warning is logged and None is returned.
        
        To use your own data directory, modify the DATA_DIR variable and ensure
        to have an "images" folder to point to your desired directory with your images.
    """
    image_path = str(DATA_DIR / "images" / filename)

    if image_path and pathlib.Path(image_path).exists():
        image = io.load_image(filepath=image_path, as_gray=as_gray, as_binary=as_binary, binary_method=binary_method)
        if image is not None:
            return image
    logger.warning(f"Image file not found: {image_path}")
    return None


def visualize_image_pair(
    input_image: np.ndarray,
    processed_images,
    title: str = "Image Processing",
):
    """
    Log one input image + multiple processed images to rerun for visualization.
    
    Note: All images are automatically converted to uint8 format for visualization
    purposes. This conversion handles various input formats (float32, float64, etc.)
    by normalizing and scaling appropriately. The original image data types are
    preserved in the actual processing results; only the visualization uses uint8.

    Args:
        input_image: The input image as a numpy array (any dtype).
        processed_images: The processed image(s) to visualize. Can be:
            - list/tuple of np.ndarray (multiple images)
            - dict[str, np.ndarray] (name -> image mapping)
            - single np.ndarray (single processed image)
        title: Title for the visualization (used in entity path naming).

    Output entity paths:
        examples/<title>/input_image (always on left)
        examples/<title>/processed_<filter_name> (on right, single image)
        examples/<title>/processed_<filter_name>_1, _2, etc. (on right, multiple images)
    """

    def _to_u8(img: np.ndarray) -> np.ndarray:
        if img is None:
            raise ValueError("image is None")
        if not isinstance(img, np.ndarray):
            raise TypeError(f"expected np.ndarray, got {type(img)}")

        if img.dtype == np.uint8:
            return img

        img = img.astype(np.float32, copy=False)
        if img.size == 0:
            return np.zeros_like(img, dtype=np.uint8)

        mx = float(np.nanmax(img))
        if mx <= 1.0:
            img = np.clip(img, 0.0, 1.0)
            return (img * 255.0).astype(np.uint8)

        # intensity-like
        return np.clip(img, 0.0, 255.0).astype(np.uint8)

    # Normalize input image
    input_image_u8 = _to_u8(input_image)

    # Extract filter name from title (remove common suffixes like "Filter", "Operation")
    filter_name = title.replace(" Filter", "").replace(" Operation", "").replace(" ", "_").lower()

    # Normalize processed collection
    if isinstance(processed_images, dict):
        items = list(processed_images.items())  # (name, img)
    elif isinstance(processed_images, (list, tuple)):
        items = [(str(i + 1) if len(processed_images) > 1 else "", img) for i, img in enumerate(processed_images)]
    else:
        # single image fallback
        items = [("", processed_images)]

    entity_base = f"examples/{title.replace(' ', '_').lower()}"

    # Log input image (always on left)
    rr.log(f"{entity_base}/input_image", rr.Image(input_image_u8))

    # Log output image(s) (always on right)
    for suffix, img in items:
        if suffix:
            # Multiple images: processed_{filter_name}_1, processed_{filter_name}_2, etc.
            output_name = f"processed_{filter_name}_{suffix}"
        else:
            # Single image: processed_{filter_name}
            output_name = f"processed_{filter_name}"
        rr.log(f"{entity_base}/{output_name}", rr.Image(_to_u8(img)))


# Filter Examples

# ===================== Contrast =====================

def enhance_image_using_clahe_example():
    """
    Applies Contrast Limited Adaptive Histogram Equalization.
    
    CLAHE enhances local contrast adaptively, preventing over-amplification
    of noise in uniform regions.
    """
    # ===================== Operation ==========================================
    
    image = load_image("brain_scan.jpg")
    logger.success("Loaded image shape: {}", image.to_numpy().shape)
    
    filtered_image = pupil.enhance_image_using_clahe(
        image=image,
        clip_limit=10.0,
        tile_grid_size=4,
        color_space="lab",
    )
    out_np_image = filtered_image.to_numpy()
    in_np_image = image.to_numpy()
    logger.success("Applied CLAHE filter. Enhanced output image shape: {}", out_np_image.shape)
    
    # ===================== Visualization  (Optional) ======================
    
    init_rerun("enhance_image_using_clahe")
    visualize_image_pair(in_np_image, out_np_image, "CLAHE Filter")

# ===================== Morphology =====================

def filter_image_using_morphological_erode_example():
    """
    Applies erosion to shrink bright regions and remove small noise.
    
    Erosion removes pixels from object boundaries, useful for removing
    small bright spots and shrinking objects.
    """
    # ===================== Operation ==========================================
    
    image = load_image("gear_with_texture.jpg", as_binary=True, binary_method='fixed')
    logger.success("Loaded image shape: {}", image.to_numpy().shape)

    filtered_image = pupil.filter_image_using_morphological_erode(
        image=image,
        kernel_size=5,
        kernel_shape="ellipse",
        iterations=10,
        border_type="default",
    )
    out_np_image = filtered_image.to_numpy()
    in_np_image = image.to_numpy()
    logger.success("Applied erosion morphological operation. Output image shape: {}", out_np_image.shape)
    
    # ===================== Visualization  (Optional) ======================
    
    init_rerun("filter_image_using_morphological_erode")
    visualize_image_pair(in_np_image, out_np_image, "Erosion Operation")


def filter_image_using_morphological_dilate_example():
    """
    Applies dilation to expand bright regions and fill holes.
    
    Dilation adds pixels to object boundaries, useful for filling gaps
    and expanding objects.
    """
    # ===================== Operation ==========================================
    
    image = load_image("spanners_arranged.jpg", as_binary=True, binary_method='fixed')
    logger.success("Loaded image shape: {}", image.to_numpy().shape)
    
    filtered_image = pupil.filter_image_using_morphological_dilate(
        image=image,
        kernel_size=5,
        kernel_shape="ellipse",
        iterations=5,
        border_type="default",
    )
    out_np_image = filtered_image.to_numpy()
    in_np_image = image.to_numpy()
    logger.success("Applied dilation morphological operation. Output image shape: {}", out_np_image.shape)
    
    # ===================== Visualization  (Optional) ======================
    
    init_rerun("filter_image_using_morphological_dilate")
    visualize_image_pair(in_np_image, out_np_image, "Dilation Operation")


def filter_image_using_morphological_close_example():
    """
    Applies a close morphological transformation.
    """
    # ===================== Operation ==========================================
    
    image = load_image("nuts_scattered.jpg", as_binary=True)
    logger.success("Loaded image shape: {}", image.to_numpy().shape)

    filtered_image = pupil.filter_image_using_morphological_close(
        image=image,
        kernel_size=5,
        kernel_shape="ellipse",
        iterations=5,
        border_type="default",
    )
    out_np_image = filtered_image.to_numpy()
    in_np_image = image.to_numpy()
    logger.success("Applied close morphological operation. Output image shape: {}", out_np_image.shape)
    
    # ===================== Visualization  (Optional) ======================
    
    init_rerun("filter_image_using_morphological_close")
    visualize_image_pair(in_np_image, out_np_image, "Closing Operation")

def filter_image_using_morphological_open_example():
    """
    Applies a open morphological transformation.
    """
    # ===================== Operation ==========================================
    
    image = load_image("broken_cables.png", as_binary=True, binary_method='fixed')
    logger.success("Loaded image shape: {}", image.to_numpy().shape)

    filtered_image = pupil.filter_image_using_morphological_open(
        image=image,
        kernel_size=3,
        kernel_shape="ellipse",
        iterations=2,
        border_type="default",
    )
    out_np_image = filtered_image.to_numpy()
    in_np_image = image.to_numpy()
    logger.success("Applied open morphological operation. Output image shape: {}", out_np_image.shape)
    
    # ===================== Visualization  (Optional) ======================
    
    init_rerun("filter_image_using_morphological_open")
    visualize_image_pair(in_np_image, out_np_image, "Opening Operation")

def filter_image_using_morphological_gradient_example():
    """
    Applies a gradient morphological transformation.
    """
    # ===================== Operation ==========================================
    
    image = load_image("cartons_arranged.png", as_gray=True)
    logger.success("Loaded image shape: {}", image.to_numpy().shape)

    filtered_image = pupil.filter_image_using_morphological_gradient(
        image=image,
        kernel_size=5,
        kernel_shape="ellipse",
        iterations=1,
        border_type="default",
    )
    out_np_image = filtered_image.to_numpy()
    in_np_image = image.to_numpy()
    logger.success("Applied gradient morphological operation. Output image shape: {}", out_np_image.shape)
    
    # ===================== Visualization  (Optional) ======================
    
    init_rerun("filter_image_using_morphological_gradient")
    visualize_image_pair(in_np_image, out_np_image, "Gradient Operation")

def filter_image_using_morphological_tophat_example():
    """
    Applies a tophat morphological transformation.
    """
    # ===================== Operation ==========================================
    
    image = load_image("keyhole.jpg", as_gray=True)
    logger.success("Loaded image shape: {}", image.to_numpy().shape)

    filtered_image = pupil.filter_image_using_morphological_tophat(
        image=image,
        kernel_size=3,
        kernel_shape="ellipse",
        iterations=5,
        border_type="default",
    )
    out_np_image = filtered_image.to_numpy()
    in_np_image = image.to_numpy()
    logger.success("Applied tophat morphological operation. Output image shape: {}", out_np_image.shape)
    
    # ===================== Visualization  (Optional) ======================
    
    init_rerun("filter_image_using_morphological_tophat")
    visualize_image_pair(in_np_image, out_np_image, "Tophat Operation")


def filter_image_using_morphological_blackhat_example():
    """
    Applies a blackhat morphological transformation.
    """
    # ===================== Operation ==========================================
    
    image = load_image("mechanical_parts_gray.png", as_gray=True)
    logger.success("Loaded image shape: {}", image.to_numpy().shape)

    filtered_image = pupil.filter_image_using_morphological_blackhat(
        image=image,
        kernel_size=15,
        kernel_shape="ellipse",
        iterations=2,
        border_type="default",
    )
    out_np_image = filtered_image.to_numpy()
    in_np_image = image.to_numpy()
    logger.success("Applied blackhat morphological operation. Output image shape: {}", out_np_image.shape)
    
    # ===================== Visualization  (Optional) ======================
    
    init_rerun("filter_image_using_morphological_blackhat")
    visualize_image_pair(in_np_image, out_np_image, "Blackhat Operation")

# ===================== Pyramid =====================

def transform_image_using_pyramid_downsampling_example():
    """
    Downsamples an image using Gaussian pyramid.
    
    Pyramid down reduces image resolution, useful for
    multi-scale analysis and efficient processing.
    """
    # ===================== Operation ==========================================
    
    image = load_image("gearbox.png")
    logger.success("Loaded image shape: {}", image.to_numpy().shape)

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
    out_np_image = filtered_image.to_numpy()
    out_np_image_1 = filtered_image_1.to_numpy()
    out_np_image_2 = filtered_image_2.to_numpy()
    in_np_image = image.to_numpy()
    logger.success("Applied pyramid downsampling. Transformed output image shapes: {}, {}, {}", out_np_image.shape, out_np_image_1.shape, out_np_image_2.shape)
    
    # ===================== Visualization  (Optional) ======================
    
    init_rerun("transform_image_using_pyramid_downsampling")
    visualize_image_pair(in_np_image, [out_np_image, out_np_image_1, out_np_image_2], "Pyramid Down")

def transform_image_using_pyramid_upsampling_example():
    """
    Upsamples an image using Gaussian pyramid.
    
    Pyramid up increases image resolution, useful for image enlargement
    and multi-scale reconstruction.
    """
    # ===================== Operation ==========================================
    
    image_in = load_image("buttons_arranged_downsampled.png")
    logger.success("Loaded image shape: {}", image_in.to_numpy().shape)

    filtered_image = pupil.transform_image_using_pyramid_upsampling(
        image=image_in,
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
    out_np_image = filtered_image.to_numpy()
    out_np_image_1 = filtered_image_1.to_numpy()
    out_np_image_2 = filtered_image_2.to_numpy()
    in_np_image = image_in.to_numpy()
    logger.success("Applied pyramid upsampling. Transformed output image shapes: {}, {}, {}", out_np_image.shape, out_np_image_1.shape, out_np_image_2.shape)
    
    # ===================== Visualization  (Optional) ======================
    
    init_rerun("transform_image_using_pyramid_upsampling")
    visualize_image_pair(in_np_image, [out_np_image, out_np_image_1, out_np_image_2], "Pyramid Up")


# ===================== Ridge / Vesselness =====================

def filter_image_using_frangi_example():
    """
    Applies Frangi vesselness filter to enhance tubular structures.
    
    Frangi filter is designed to detect vessel-like structures in medical
    images, fingerprints, and other images with elongated features.
    """
    # ===================== Operation ==========================================

    image = load_image("tablets_arranged.jpg", as_gray=True)
    logger.success("Loaded image shape: {}", image.to_numpy().shape)

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
    out_np_image = filtered_image.to_numpy()
    in_np_image = image.to_numpy()
    logger.success("Applied Frangi filter. Filtered output image shape: {}", out_np_image.shape)

    # ===================== Visualization  (Optional) ======================
    
    init_rerun("filter_image_using_frangi")
    visualize_image_pair(in_np_image, out_np_image, "Frangi Filter")


def filter_image_using_hessian_example():
    """
    Applies Hessian-based vesselness filter for tubular structure detection.
    
    Hessian filter uses eigenvalue analysis to detect vessel-like structures,
    similar to Frangi but with different vesselness measure.
    """
    # ===================== Operation ==========================================
    
    image = load_image("wires.jpg", as_gray=True)
    logger.success("Loaded image shape: {}", image.to_numpy().shape)

    filtered_image = pupil.filter_image_using_hessian(
        image=image,
        scale_start=1,
        scale_end=6,
        scale_step=1,
        detect_black_ridges=True,
        border_type="reflect",
        constant_value=0.0,
    )

    out_np_image = filtered_image.to_numpy()
    in_np_image = image.to_numpy()
    logger.success("Applied Hessian filter. Filtered output image shape: {}", out_np_image.shape)

    # ===================== Visualization  (Optional) ======================
    
    init_rerun("filter_image_using_hessian")
    visualize_image_pair(in_np_image, out_np_image, "Hessian Filter")


def filter_image_using_sato_example():
    """
    Applies Sato filter for multi-scale ridge detection.
    
    Sato filter is designed to detect ridges and valleys at multiple scales,
    useful for detecting fine structures like vessels or fibers.
    """
    # ===================== Operation ==========================================
    
    image = load_image("pcb_top_gray.png", as_gray=True)
    logger.success("Loaded image shape: {}", image.to_numpy().shape)

    filtered_image = pupil.filter_image_using_sato(
        image=image,
        scale_start=1,
        scale_end=12,
        scale_step=1,
        detect_black_ridges=False,
        border_type="reflect",
        constant_value=0.0,
    )
    out_np_image = filtered_image.to_numpy()
    in_np_image = image.to_numpy()
    logger.success("Applied Sato filter. Filtered output image shape: {}", out_np_image.shape)

    # ===================== Visualization  (Optional) ======================
    
    init_rerun("filter_image_using_sato")
    visualize_image_pair(in_np_image, out_np_image, "Sato Filter")


def filter_image_using_meijering_example():
    """
    Applies Meijering filter for neurite detection.
    
    Meijering filter is optimized for detecting neurites and similar
    branching structures in biomedical images.
    """
    # ===================== Operation ==========================================
    
    image = load_image("sidewalk_cracked.jpg", as_gray=True)
    logger.success("Loaded image shape: {}", image.to_numpy().shape)  

    filtered_image = pupil.filter_image_using_meijering(
        image=image,
        scale_start=1,
        scale_end=10,
        scale_step=2,
        detect_black_ridges=True,
        border_type="reflect",
        constant_value=0.0,
    )
    out_np_image = filtered_image.to_numpy()
    in_np_image = image.to_numpy()
    logger.success("Applied Meijering filter. Filtered output image shape: {}", out_np_image.shape)

    # ===================== Visualization  (Optional) ======================
    
    init_rerun("filter_image_using_meijering")
    visualize_image_pair(in_np_image, out_np_image, "Meijering Filter")

# ===================== Sharpening / Gradients =====================

def filter_image_using_laplacian_example():
    """
    Applies Laplacian filter for edge detection using second derivatives.
    
    Laplacian operator detects edges by finding regions where the second
    derivative is zero or changes sign.
    """
    # ===================== Operation ==========================================
    
    image = load_image("flat_mechanical_component_denoised.png")
    logger.success("Loaded image shape: {}", image.to_numpy().shape)

    filtered_image = pupil.filter_image_using_laplacian(
        image=image,
        kernel_size=5,
        scale=1.0,
        delta=0.0,
        output_format="32bit",
        border_type="default",
    )
    out_np_image = filtered_image.to_numpy()
    in_np_image = image.to_numpy()
    logger.success("Applied Laplacian filter. Filtered output image shape: {}", out_np_image.shape)
    
    # ===================== Visualization  (Optional) ======================
    
    init_rerun("filter_image_using_laplacian")
    visualize_image_pair(in_np_image, out_np_image, "Laplacian Filter")


def filter_image_using_sobel_example():
    """
    Applies Sobel filter for directional edge detection.
    
    Sobel operator computes gradients in X and Y directions, useful for
    detecting edges and their orientation.
    """
    # ===================== Operation ==========================================
    
    image = load_image("nuts_scattered.jpg", as_gray=True)
    logger.success("Loaded image shape: {}", image.to_numpy().shape)

    filtered_image = pupil.filter_image_using_sobel(
        image=image,
        dx=1,
        dy=1,
        kernel_size=9,
        scale=1.0,
        delta=0.0,
        output_format="64bit",
        border_type="default",
    )
    out_np_image = filtered_image.to_numpy()
    in_np_image = image.to_numpy()
    logger.success("Applied Sobel filter. Filtered output image shape: {}", out_np_image.shape)
    
    # ===================== Visualization  (Optional) ======================
    
    init_rerun("filter_image_using_sobel")
    visualize_image_pair(in_np_image, out_np_image, "Sobel Filter")


def filter_image_using_scharr_example():
    """
    Applies Scharr filter for improved edge detection accuracy.
    
    Scharr operator is similar to Sobel but with better rotation invariance
    and more accurate gradient estimation.
    """
    # ===================== Operation ==========================================
    
    image = load_image("nuts_scattered.jpg", as_gray=True)
    logger.success("Loaded image shape: {}", image.to_numpy().shape)

    filtered_image = pupil.filter_image_using_scharr(
        image=image,
        dx=0,
        dy=1,
        scale=1.0,
        delta=0.0,
        output_format="8bit",
        border_type="default",
    )
    out_np_image = filtered_image.to_numpy()
    in_np_image = image.to_numpy()
    logger.success("Applied Scharr filter. Filtered output image shape: {}", out_np_image.shape)

    # ===================== Visualization  (Optional) ======================
    
    init_rerun("filter_image_using_scharr")
    visualize_image_pair(in_np_image, out_np_image, "Scharr Filter")

def filter_image_using_gabor_example():
    """
    Applies Gabor filter for texture analysis and feature detection.
    
    Gabor filters are useful for detecting oriented features and textures
    at specific scales and orientations.
    """
    # ===================== Operation ==========================================
    
    image = load_image("finger_print.jpg", as_gray=True)
    logger.success("Loaded image shape: {}", image.to_numpy().shape)
    
    filtered_image = pupil.filter_image_using_gabor(
        image=image,
        kernel_size=5,
        standard_deviation=15.0,
        orientation=0.0,
        wavelength=10.0,
        aspect_ratio=0.5,
        phase_offset=np.pi * 0.5,
    )
    out_np_image = filtered_image.to_numpy()
    in_np_image = image.to_numpy()
    logger.success("Applied Gabor filter. Filtered output image shape: {}", out_np_image.shape)
    
    # ===================== Visualization  (Optional) ======================
    
    if out_np_image.dtype != np.uint8:
        # Handle float images like cv2.imshow does
        if out_np_image.dtype in [np.float32, np.float64]:
            # Clip to [0, 1] range
            out_np_image = np.clip(out_np_image, 0, 1)
            # Scale to [0, 255]
            out_np_image = (out_np_image * 255).astype(np.uint8)
    
    init_rerun("filter_image_using_gabor")
    visualize_image_pair(in_np_image, out_np_image, "Gabor Filter")


# ===================== Smoothing =====================

def filter_image_using_bilateral_example():
    """
    Applies a bilateral filter to reduce noise while preserving edges.
    
    Bilateral filtering is effective for noise reduction while maintaining
    edge sharpness. It considers both spatial proximity and color similarity.
    """
    # ===================== Operation ==========================================

    image = load_image("nuts_scattered_noised.jpg")
    logger.success("Loaded image shape: {}", image.to_numpy().shape)

    filtered_image = pupil.filter_image_using_bilateral(
        image=image,
        neighborhood_diameter=19,
        spatial_sigma=75.0,
        color_intensity_sigma=100.0,
        border_type="default",
    )  
    out_np_image = filtered_image.to_numpy()
    in_np_image = image.to_numpy()
    logger.success("Applied Bilateral filter. Filtered output image shape: {}", out_np_image.shape)
    
    # ===================== Visualization  (Optional) ======================
    
    init_rerun("filter_image_using_bilateral")
    visualize_image_pair(in_np_image, out_np_image, "Bilateral Filter")


def filter_image_using_median_blur_example():
    """
    Applies median blur to reduce salt-and-pepper noise.
    
    Median blur replaces each pixel with the median of its neighborhood,
    effectively removing impulse noise while preserving edges.
    """
    # ===================== Operation ==========================================
    
    image = load_image("flat_mechanical_component.png")
    logger.success("Loaded image shape: {}", image.to_numpy().shape)
    filtered_image = pupil.filter_image_using_median_blur(
        image=image,
        kernel_size=11,
    )
    out_np_image = filtered_image.to_numpy()
    in_np_image = image.to_numpy()
    logger.success("Applied Median Blur filter. Filtered output image shape: {}", out_np_image.shape)
    
    # ===================== Visualization  (Optional) ======================
    
    init_rerun("filter_image_using_median_blur")
    visualize_image_pair(in_np_image, out_np_image, "Median Blur")


def filter_image_using_box_example():
    """
    Applies a normalized box filter with configurable depth and normalization.
    
    Box filter performs normalized averaging within a kernel region.
    Useful for basic smoothing operations.
    """
    # ===================== Operation ==========================================
    
    image = load_image("nuts_scattered_noised.jpg")
    logger.success("Loaded image shape: {}", image.to_numpy().shape)

    filtered_image = pupil.filter_image_using_box(
        image=image,
        kernel_size=5,
        normalize=True,
        output_format="8bit",
        border_type="reflect",
    )
    out_np_image = filtered_image.to_numpy()
    in_np_image = image.to_numpy()
    logger.success("Applied Box filter. Filtered output image shape: {}", out_np_image.shape)
    
    # ===================== Visualization  (Optional) ======================
    
    init_rerun("filter_image_using_box")
    visualize_image_pair(in_np_image, out_np_image, "Box Filter")


def filter_image_using_gaussian_blur_example():
    """
    Applies Gaussian blur for smooth noise reduction.
    
    Gaussian blur uses a Gaussian kernel for weighted averaging, providing
    natural-looking blur with better edge preservation than simple blur.
    """
    # ===================== Operation ==========================================
    
    image = load_image("nuts_scattered_noised.jpg")
    logger.success("Loaded image shape: {}", image.to_numpy().shape)

    filtered_image = pupil.filter_image_using_gaussian_blur(
        image=image,
        kernel_size=19,
        sigma_x=2.0,
        sigma_y=3.0,
        border_type="default",
    )
    out_np_image = filtered_image.to_numpy()
    in_np_image = image.to_numpy()
    logger.success("Applied Gaussian Blur filter. Filtered output image shape: {}", out_np_image.shape)
    
    # ===================== Visualization  (Optional) ======================
    
    init_rerun("filter_image_using_gaussian_blur")
    visualize_image_pair(in_np_image, out_np_image, "Gaussian Blur")


def filter_image_using_blur_example():
    """
    Simple average blur.
    """
    # ===================== Operation ==========================================
    
    image = load_image("nuts_scattered_noised.jpg")
    logger.success("Loaded image shape: {}", image.to_numpy().shape)

    filtered_image = pupil.filter_image_using_blur(
        image=image,
        kernel_size=7,
        border_type="default",
    )
    out_np_image = filtered_image.to_numpy()
    in_np_image = image.to_numpy()
    logger.success("Applied Blur filter. Filtered output image shape: {}", out_np_image.shape)
    
    # ===================== Visualization  (Optional) ======================
    
    init_rerun("filter_image_using_blur")
    visualize_image_pair(in_np_image, out_np_image, "Blur")


def enhance_image_using_auto_gamma_correction_example():
    """
    Applies gamma correction to adjust image brightness non-linearly.
    """
    # ===================== Operation ==========================================
    
    image = load_image("screws_in_dark_lighting.jpg")
    logger.success("Loaded image shape: {}", image.to_numpy().shape)

    filtered_image = pupil.enhance_image_using_auto_gamma_correction(
        image=image,
    )
    out_np_image = filtered_image.to_numpy()
    in_np_image = image.to_numpy()
    logger.success("Applied Auto Gamma Correction. Enhanced output image shape: {}", out_np_image.shape)
    
    # ===================== Visualization  (Optional) ======================
    
    init_rerun("enhance_image_using_auto_gamma_correction_example")
    visualize_image_pair(in_np_image, out_np_image, "Gamma Correction")


def enhance_image_using_white_balance_example():
    """
    White balance (simple per-channel scaling).
    """
    # ===================== Operation ==========================================
    
    image = load_image("hand_tools_yellow_light.png")
    logger.success("Loaded image shape: {}", image.to_numpy().shape)

    filtered_image = pupil.enhance_image_using_white_balance(
        image=image,
    )
    out_np_image = filtered_image.to_numpy()
    in_np_image = image.to_numpy()
    logger.success("Applied White Balance. Enhanced output image shape: {}", out_np_image.shape)
    
    # ===================== Visualization  (Optional) ======================
    
    init_rerun("enhance_image_using_white_balance")
    visualize_image_pair(in_np_image, out_np_image, "White Balance")

# ===================== Thinning =====================

def transform_mask_using_blob_thinning_example():
    """
    Skeletonizes (thins) foreground blobs in a binary mask.
    """
    # ===================== Operation ==========================================
    
    image = load_image("handwriting_mask.png", as_binary=True)
    logger.success("Loaded image shape: {}", image.to_numpy().shape)

    filtered_image = pupil.transform_mask_using_blob_thinning(
        image=image,
        thinning_type="thinning_zhangsuen",
    )
    out_np_image = filtered_image.to_numpy()
    in_np_image = image.to_numpy()
    logger.success("Applied Zhang-Suen thinning. Transformed output image shape 1: {}", out_np_image.shape)
    
    # ===================== Visualization  (Optional) ======================
    
    init_rerun("transform_mask_using_blob_thinning")
    visualize_image_pair(in_np_image, out_np_image, "Thinning Filter 1")

    # ===================== Operation ==========================================
    
    image = load_image("male_female_mask.png", as_binary=True)

    filtered_image = pupil.transform_mask_using_blob_thinning(
        image=image,
        thinning_type="thinning_zhangsuen",
    )
    out_np_image = filtered_image.to_numpy()
    in_np_image = image.to_numpy()
    logger.success("Applied Zhang-Suen thinning. Transformed output image shape 2: {}", out_np_image.shape)
    
    # ===================== Visualization  (Optional) ======================
    
    visualize_image_pair(in_np_image, out_np_image, "Thinning Filter 2")


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

