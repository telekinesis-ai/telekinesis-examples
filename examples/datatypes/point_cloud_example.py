"""Demonstrates the Telekinesis PointCloud datatype."""

import time
from pathlib import Path

import numpy as np
import rerun as rr
from loguru import logger

from telekinesis import datatypes

def point_cloud_example():
    """Demonstrate creation, inspection, operations, visualization, and serialization."""

    # ======================= Create ============================================
    N = 2000
    positions = np.random.randn(N, 3).astype(np.float32)
    point_cloud = datatypes.PointCloud(positions)
    logger.info(f"Created PointCloud: {point_cloud}")

    normals = np.random.randn(N, 3).astype(np.float32)
    colors = np.random.randint(0, 255, (N, 3), dtype=np.uint8)
    compression = datatypes.PointCloudCompression.DRACO
    point_cloud = datatypes.PointCloud(
        positions, normals=normals, colors=colors, compression=compression
    )
    logger.info(f"PointCloud with normals, colors, and compression: {point_cloud}")

    url = "https://assets.telekinesis.ai/examples/v1/point_clouds/zivid_bottles_10_preprocessed.ply"
    point_cloud_from_url = datatypes.PointCloud.from_url(url=url)
    logger.info(f"PointCloud loaded from URL: {point_cloud_from_url}")

    cached_path = Path.home() / ".cache" / "telekinesis" / "point_clouds" / Path(url).name
    point_cloud_from_path = datatypes.PointCloud.from_path(cached_path)
    logger.info(f"PointCloud loaded from path: {point_cloud_from_path}")

    # ======================= Inspect ===========================================
    logger.info(f"positions={point_cloud.positions}")
    logger.info(f"normals={point_cloud.normals}")
    logger.info(f"colors={point_cloud.colors}")
    logger.info(f"has_normals={point_cloud.has_normals}")
    logger.info(f"has_colors={point_cloud.has_colors}")
    logger.info(f"use_compression={point_cloud.compression}")
    logger.info(f"compression_settings={point_cloud.compression_settings}")
    logger.info(f"draco_atol={point_cloud.draco_atol}")

    # ======================= Operations =========================================
    point_cloud.positions = np.random.randn(N, 3).astype(np.float32)
    logger.info(f"Updated positions: {point_cloud}")

    point_cloud.normals = np.random.randn(N, 3).astype(np.float32)
    logger.info(f"Updated normals: {point_cloud}")

    point_cloud.colors = np.random.randint(0, 255, (N, 3), dtype=np.uint8)
    logger.info(f"Updated colors: {point_cloud}")

    point_cloud.set_compression_parameters(compression_level=5, quantization_bits=12)
    logger.info(f"Updated compression settings: {point_cloud.compression_settings}")

    point_cloud_copy = point_cloud.copy()
    logger.info(f"Copied PointCloud: {point_cloud_copy}")

    point_cloud_numpy = point_cloud.to_numpy(copy=True)
    logger.info(f"NumPy positions:\n{point_cloud_numpy}")

    array_data = np.asarray(point_cloud)
    centroid = np.mean(point_cloud, axis=0)
    logger.info(f"As array: {array_data}")
    logger.info(f"Centroid: {centroid}")

    logger.info(f"length={len(point_cloud)}")

    save_path = "results/point_cloud_example.ply"
    point_cloud.save_to_path(save_path)
    logger.info(f"Saved PointCloud to {save_path}")

    # ======================= Visualize =========================================
    rr.init("point_cloud_example", spawn=True)
    datatypes.visualize(
        point_cloud, entity_path="/point_cloud/updated", label="Updated PointCloud"
    )
    datatypes.visualize(
        point_cloud_from_url, entity_path="/point_cloud/from_url", label="URL PointCloud"
    )

    # ======================= Serialize / Deserialize ===========================
    start = time.perf_counter()
    serialized = datatypes.serialize(point_cloud)
    serialization_ms = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    deserialized = datatypes.deserialize(serialized)["param_0"]
    deserialization_ms = (time.perf_counter() - start) * 1000

    logger.info(f"Deserialized PointCloud: {deserialized}")
    logger.info(f"Round-trip successful: {point_cloud == deserialized}")
    logger.info(f"Serialization time: {serialization_ms:.3f} ms")
    logger.info(f"Deserialization time: {deserialization_ms:.3f} ms")


if __name__ == "__main__":
    point_cloud_example()
