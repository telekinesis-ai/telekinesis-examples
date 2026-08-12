"""
Example script to demonstrate usage of DepthImage datatype.
"""

import time

import numpy as np
from loguru import logger
import rerun as rr

from telekinesis import datatypes


def depth_image_example():
    """
    Example function to demonstrate usage of DepthImage datatype.
        - Create a DepthImage data
        - Access the underlying depth data
        - Visualize the DepthImage data using Rerun
        - Update the underlying depth data
        - Create a DepthImage with an aligned color image (RGB-D)
        - Create a DepthImage with ZSTD compression
        - Operate DepthImage with numpy
        - Serialize and deserialize the depth image
    """
    H, W = 480, 640

    # Create a DepthImage data
    input_depth = (np.random.rand(H, W) * 5.0).astype(np.float32)
    logger.info(f"Input Depth shape: {input_depth.shape}, dtype: {input_depth.dtype}")
    my_depth_image = datatypes.DepthImage(input_depth)
    logger.info(f"Original DepthImage: {my_depth_image}")

    # Access the underlying depth data
    my_depth_data = my_depth_image.depth
    my_depth_shape = my_depth_image.shape
    my_depth_height = my_depth_image.height
    my_depth_width = my_depth_image.width
    my_depth_has_colors = my_depth_image.has_colors
    my_depth_compression = my_depth_image.compression
    my_depth_numpy_array = np.asarray(my_depth_image)

    logger.info(f"Underlying Depth data: {my_depth_data}")
    logger.info(f"Underlying Depth shape: {my_depth_shape}")
    logger.info(f"Underlying Depth height: {my_depth_height}")
    logger.info(f"Underlying Depth width: {my_depth_width}")
    logger.info(f"Underlying Depth has_colors: {my_depth_has_colors}")
    logger.info(f"Underlying Depth compression: {my_depth_compression}")
    logger.info(f"Underlying Depth numpy array: {my_depth_numpy_array}")

    logger.info("Visualizing with Rerun...")
    rr.init("depth_image_example", spawn=True)
    datatypes.visualize(my_depth_image, entity_path="/DepthImage")

    # Update the depth data.
    # DepthImage has no in-place setter (unlike Image.data), so updating
    # means constructing a new instance with fresh depth values.
    updated_depth = (np.random.rand(H, W) * 5.0).astype(np.float32)
    my_depth_image = datatypes.DepthImage(updated_depth)
    logger.info("Visualize updated depth image with entity path /DepthImage...")
    datatypes.visualize(my_depth_image, entity_path="/DepthImage")

    # Create a DepthImage with an aligned color image (RGB-D)
    colors = np.random.randint(0, 255, (H, W, 3), dtype=np.uint8)
    rgbd_image = datatypes.DepthImage(input_depth, colors=colors)
    logger.info(f"My RGB-D DepthImage: {rgbd_image}")
    datatypes.visualize(rgbd_image, entity_path="/RGBDImage")

    # Create a DepthImage with ZSTD compression
    zstd_image = datatypes.DepthImage(
        input_depth,
        colors=colors,
        compression=datatypes.ImageCompression.ZSTD,
    )
    logger.info(f"My ZSTD DepthImage: {zstd_image}")
    datatypes.visualize(zstd_image, entity_path="/ZSTDImage")

    # Operate DepthImage with numpy
    my_depth_mean = np.mean(my_depth_image)
    logger.info(f"Mean depth value: {my_depth_mean}")

    my_depth_flipped = np.flipud(my_depth_image)
    logger.info(f"Flipped Depth shape: {my_depth_flipped.shape}, dtype: {my_depth_flipped.dtype}")

    # Serialize and deserialize the depth image
    serialization_start_time = time.perf_counter()
    serialized_depth_image = datatypes.serialize(rgbd_image)
    serialization_end_time = time.perf_counter()

    deserialization_start_time = time.perf_counter()
    deserialized_depth_image = datatypes.deserialize(serialized_depth_image)["param_0"]
    deserialization_end_time = time.perf_counter()
    logger.info(
        f"Deserialize DepthImage matches with original: {deserialized_depth_image == rgbd_image}"
    )

    logger.info(
        f"Serialization time: {(serialization_end_time - serialization_start_time) * 1000} ms"
    )
    logger.info(
        f"Deserialization time: {(deserialization_end_time - deserialization_start_time) * 1000} ms"
    )


if __name__ == "__main__":
    depth_image_example()
