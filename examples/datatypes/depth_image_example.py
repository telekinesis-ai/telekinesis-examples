"""Demonstrates the Telekinesis DepthImage datatype."""

import time

import numpy as np
from loguru import logger
import rerun as rr

from telekinesis import datatypes

def depth_image_example():
    """Demonstrate creation, inspection, visualization, update, RGB-D, compression, NumPy interop, and serialization."""

    # ======================= Create ============================================
    H, W = 480, 640
    depth = (np.random.rand(H, W) * 5.0).astype(np.float32)
    depth_image = datatypes.DepthImage(depth)

    logger.info(f"Input depth shape={depth.shape}, dtype={depth.dtype}")
    logger.info(f"Original DepthImage: {depth_image}")

    # ======================= Inspect ===========================================
    data = depth_image.depth
    shape = depth_image.shape
    height = depth_image.height
    width = depth_image.width
    has_colors = depth_image.has_colors
    compression = depth_image.compression
    numpy_array = np.asarray(depth_image)

    logger.info(
        f"shape={shape}, "
        f"height={height}, "
        f"width={width}, "
        f"has_colors={has_colors}, "
        f"compression={compression}"
    )
    logger.info(f"Data: {data}")
    logger.info(f"NumPy array: {numpy_array}")

    # ======================= Visualize =========================================
    rr.init("depth_image_example", spawn=True)
    datatypes.visualize(depth_image, entity_path="/DepthImage")

    # ======================= Update ============================================
    new_depth = (np.random.rand(H, W) * 5.0).astype(np.float32)
    depth_image = datatypes.DepthImage(new_depth)
    datatypes.visualize(depth_image, entity_path="/DepthImage")

    # ======================= RGB-D =============================================
    colors = np.random.randint(0, 255, (H, W, 3), dtype=np.uint8)
    rgbd_image = datatypes.DepthImage(depth, colors=colors)
    logger.info(f"RGB-D DepthImage: {rgbd_image}")
    datatypes.visualize(rgbd_image, entity_path="/RGBDImage")

    # ======================= ZSTD Compression ==================================
    zstd_image = datatypes.DepthImage(
        depth,
        colors=colors,
        compression=datatypes.ImageCompression.ZSTD,
    )
    logger.info(f"ZSTD DepthImage: {zstd_image}")
    datatypes.visualize(zstd_image, entity_path="/ZSTDImage")

    # ======================= NumPy Interop =====================================
    mean_depth = np.mean(depth_image)
    flipped_depth = np.flipud(depth_image)

    logger.info(f"Mean depth value: {mean_depth}")
    logger.info(f"Flipped depth shape={flipped_depth.shape}, dtype={flipped_depth.dtype}")

    # ======================= Serialize / Deserialize ===========================
    start = time.perf_counter()
    serialized = datatypes.serialize(rgbd_image)
    serialization_ms = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    deserialized = datatypes.deserialize(serialized)["param_0"]
    deserialization_ms = (time.perf_counter() - start) * 1000

    logger.info(f"Deserialized DepthImage: {deserialized}")
    logger.info(f"Round-trip successful: {deserialized == rgbd_image}")
    logger.info(f"Serialization time: {serialization_ms:.3f} ms")
    logger.info(f"Deserialization time: {deserialization_ms:.3f} ms")


if __name__ == "__main__":
    depth_image_example()
