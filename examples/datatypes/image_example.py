"""
Example script to demonstrate usage of Image datatype.
"""

import time
from pathlib import Path

import numpy as np
import requests
from loguru import logger
import rerun as rr

from telekinesis import datatypes


def image_example():
    """
    Example function to demonstrate usage of Image datatype.
        - Create an Image data
        - Access the underlying image data
        - Visualize the Image data using Rerun
        - Update the underlying image data
        - Create image from a path
        - Create image from a URL
        - Convert to gray scale
        - Convert to RGB
        - Convert to BGR
        - Expand dimensions of the image
        - Save the image to a path
        - Operate Image with numpy
        - Serialize and deserialize the image
    """
    # Create an Image data
    input_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    logger.info(f"Input Image shape: {input_image.shape}, dtype: {input_image.dtype}")
    my_image = datatypes.Image(input_image)
    logger.info(f"Original Image: {my_image}")

    # Access the underlying image data
    my_image_data = my_image.data
    my_image_shape = my_image_data.shape
    my_image_dtype = my_image_data.dtype
    my_image_height = my_image.height
    my_image_width = my_image.width
    my_image_channels = my_image.channels
    my_image_compression = my_image.compression
    my_image_numpy_array = my_image.to_numpy()

    logger.info(f"Underlying Image data: {my_image_data}")
    logger.info(f"Underlying Image shape: {my_image_shape}")
    logger.info(f"Underlying Image dtype: {my_image_dtype}")
    logger.info(f"Underlying Image height: {my_image_height}")
    logger.info(f"Underlying Image width: {my_image_width}")
    logger.info(f"Underlying Image channels: {my_image_channels}")
    logger.info(f"Underlying Image compression: {my_image_compression}")
    logger.info(f"Underlying Image numpy array: {my_image_numpy_array}")

    logger.info("Visualizing with Rerun...")
    rr.init("image_example", spawn=True)
    datatypes.visualize(my_image, entity_path="/Image")

    # Update the image data
    updated_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    my_image.data = updated_image
    logger.info("Visualize updated image with entity path /Image...")
    datatypes.visualize(my_image, entity_path="/Image")

    # Create image from a path
    ROOT_PATH = Path(__file__).parent
    my_image_from_path = datatypes.Image.from_path(ROOT_PATH / "data/sample.jpg")
    logger.info(
        f"New image from path shape: {my_image_from_path.shape}, dtype: {my_image_from_path.dtype}"
    )
    datatypes.visualize(my_image_from_path, entity_path="/ImageFromPath")

    # Create image from URL
    image_url = "https://telekinesis-public-assets.s3.us-east-1.amazonaws.com/examples/v1/images/screws_standing.jpg"
    response = requests.get(image_url, timeout=60)
    response.raise_for_status()
    my_image_url = datatypes.Image.from_encoded_buffer(response.content)
    logger.info(f"New image from URL shape: {my_image_url.shape}, dtype: {my_image_url.dtype}")
    datatypes.visualize(my_image_url, entity_path="/ImageFromURL")

    # Convert to gray scale
    gray_image = my_image_from_path.to_grayscale()
    logger.info(f"Gray Image shape: {gray_image.shape}, dtype: {gray_image.dtype}")
    datatypes.visualize(gray_image, entity_path="/GrayImage")

    # Convert to BGR
    bgr_image = my_image_from_path.to_bgr()
    logger.info(f"BGR Image shape: {bgr_image.shape}, dtype: {bgr_image.dtype}")
    datatypes.visualize(bgr_image, entity_path="/BGRImage")

    # Convert to back to RGB
    rgb_image = bgr_image.to_rgb()
    logger.info(f"RGB Image shape: {rgb_image.shape}, dtype: {rgb_image.dtype}")
    datatypes.visualize(rgb_image, entity_path="/RGBImage")
    # Expand dimensions of the image
    # This returns a new ImageBatch object.
    expanded_image = my_image.expand_dims()
    logger.info(f"Expanded Image shapes: {expanded_image.shapes}, dtypes: {expanded_image.dtypes}")

    # Save the image to a path
    output_image_path = ROOT_PATH / "data/output_image.jpg"
    gray_image.save_to_path(path=output_image_path)
    logger.info(f"Image saved to: {output_image_path}")

    # Operate Image with numpy
    my_image_mean = np.mean(my_image)
    logger.info(f"Mean pixel value of the image: {my_image_mean}")

    my_image_flipped = np.flipud(my_image)
    logger.info(f"Flipped Image shape: {my_image_flipped.shape}, dtype: {my_image_flipped.dtype}")

    # Serialize and deserialize the image
    serialization_start_time = time.perf_counter()
    serialized_image = datatypes.serialize(my_image)
    serialization_end_time = time.perf_counter()

    deserialization_start_time = time.perf_counter()
    deserialized_image = datatypes.deserialize(serialized_image)["param_0"]
    deserialization_end_time = time.perf_counter()
    logger.info(f"Deserialize Image matches with original: {deserialized_image == my_image}")

    logger.info(
        f"Serialization time: {(serialization_end_time - serialization_start_time) * 1000} ms"
    )
    logger.info(
        f"Deserialization time: {(deserialization_end_time - deserialization_start_time) * 1000} ms"
    )


if __name__ == "__main__":
    image_example()
