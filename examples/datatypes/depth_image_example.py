"""Demonstrates the Telekinesis DepthImage datatype."""

import time
from pathlib import Path

import numpy as np
import rerun as rr
from loguru import logger

from telekinesis import datatypes

def depth_image_example():
    """Demonstrate creation, inspection, operations, visualization, and serialization."""

    # ======================= Create ============================================
    H, W = 480, 640
    depth = (np.random.rand(H, W) * 5.0).astype(np.float32)
    depth_image = datatypes.DepthImage(depth)
    logger.info(f"Created DepthImage: {depth_image}")

    depth_image_from_coerce = datatypes.DepthImage.coerce(depth)
    logger.info(f"DepthImage created via coerce: {depth_image_from_coerce}")

    depth_scale = 0.001
    raw_counts = np.round(depth / depth_scale).astype(np.uint16)
    depth_image_from_raw_buffer = datatypes.DepthImage.from_raw_buffer(
        raw_counts.tobytes(), shape=depth.shape, dtype=raw_counts.dtype, depth_scale=depth_scale
    )
    logger.info(f"DepthImage created from raw buffer: {depth_image_from_raw_buffer}")

    save_path = Path("results/depth_image_example.png")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    depth_image.save_to_path(save_path, depth_scale=depth_scale)
    depth_image_from_path = datatypes.DepthImage.from_path(save_path, depth_scale=depth_scale)
    logger.info(f"DepthImage created from path: {depth_image_from_path}")

    encoded_buffer = save_path.read_bytes()
    depth_image_from_encoded_buffer = datatypes.DepthImage.from_encoded_buffer(
        encoded_buffer, depth_scale=depth_scale
    )
    logger.info(f"DepthImage created from encoded buffer: {depth_image_from_encoded_buffer}")

    # ======================= Inspect ===========================================
    logger.info(f"depth={depth_image.depth}")
    logger.info(f"colors={depth_image.colors}")
    logger.info(f"shape={depth_image.shape}")
    logger.info(f"height={depth_image.height}")
    logger.info(f"width={depth_image.width}")
    logger.info(f"has_colors={depth_image.has_colors}")
    logger.info(f"compression={depth_image.compression}")

    # ======================= Operations =========================================
    colors = np.random.randint(0, 255, (H, W, 3), dtype=np.uint8)
    rgbd_image = datatypes.DepthImage(depth, colors=colors)
    logger.info(f"RGB-D DepthImage: {rgbd_image}")

    zstd_image = datatypes.DepthImage(
        depth, colors=colors, compression=datatypes.ImageCompression.ZSTD
    )
    logger.info(f"ZSTD-compressed DepthImage: {zstd_image}")

    depth_image_copy = depth_image.copy()
    logger.info(f"Copied DepthImage: {depth_image_copy}")

    depth_image_numpy = depth_image.to_numpy(copy=True)
    logger.info(f"NumPy depth array: {depth_image_numpy}")

    numpy_array = np.asarray(depth_image)
    logger.info(f"Mean depth value: {np.mean(numpy_array)}")
    logger.info(f"Flipped depth shape: {np.flipud(numpy_array).shape}")

    # ======================= Visualize =========================================
    rr.init("depth_image_example", spawn=True)
    datatypes.visualize(depth_image, entity_path="/depth_image/original")
    datatypes.visualize(rgbd_image, entity_path="/depth_image/rgbd")
    datatypes.visualize(zstd_image, entity_path="/depth_image/zstd")

    # ======================= Serialize / Deserialize ===========================
    start = time.perf_counter()
    serialized = datatypes.serialize(rgbd_image)
    serialization_ms = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    deserialized = datatypes.deserialize(serialized)["param_0"]
    deserialization_ms = (time.perf_counter() - start) * 1000

    logger.info(f"Deserialized DepthImage: {deserialized}")
    logger.info(f"Round-trip successful: {rgbd_image == deserialized}")
    logger.info(f"Serialization time: {serialization_ms:.3f} ms")
    logger.info(f"Deserialization time: {deserialization_ms:.3f} ms")


if __name__ == "__main__":
    depth_image_example()
