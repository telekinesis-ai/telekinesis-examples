"""Demonstrates the Telekinesis PointCloudBatch datatype."""

import time

import numpy as np
import rerun as rr
from loguru import logger

from telekinesis import datatypes

def point_cloud_batch_example():
    """Demonstrate creation, inspection, operations, visualization, and serialization."""

    # ======================= Create ============================================
    N = 2000
    cloud_1 = datatypes.PointCloud(
        np.random.randn(N, 3).astype(np.float32) + np.array([-5.0, 0.0, 0.0], dtype=np.float32),
        normals=np.random.randn(N, 3).astype(np.float32),
        colors=np.random.randint(0, 255, (N, 3), dtype=np.uint8),
    )
    cloud_2 = np.random.randn(N, 3).astype(np.float32) + np.array(
        [5.0, 0.0, 0.0], dtype=np.float32
    )
    point_cloud_batch = datatypes.PointCloudBatch([cloud_1, cloud_2])
    logger.info(f"Created PointCloudBatch: {point_cloud_batch}")

    # ======================= Inspect ===========================================
    logger.info(f"positions={point_cloud_batch.positions}")
    logger.info(f"normals={point_cloud_batch.normals}")
    logger.info(f"colors={point_cloud_batch.colors}")
    logger.info(f"length={len(point_cloud_batch)}")

    # ======================= Operations =========================================
    single_cloud = point_cloud_batch[0]
    logger.info(f"Single PointCloud at index 0: {single_cloud}")

    sliced_batch = point_cloud_batch[0:1]
    logger.info(f"Sliced PointCloudBatch: {sliced_batch}")

    mask = np.array([True, False])
    masked_batch = point_cloud_batch[mask]
    logger.info(f"Masked PointCloudBatch: {masked_batch}")

    point_cloud_batch_copy = point_cloud_batch.copy()
    logger.info(f"Copied PointCloudBatch: {point_cloud_batch_copy}")

    point_cloud_batch_numpy = point_cloud_batch.to_numpy(copy=True)
    logger.info(f"NumPy positions per cloud: {[array.shape for array in point_cloud_batch_numpy]}")

    updated_cloud = datatypes.PointCloud(
        np.random.randn(N, 3).astype(np.float32),
        colors=np.full((N, 3), [255, 0, 0], dtype=np.uint8),
    )
    rebuilt_batch = datatypes.PointCloudBatch([cloud_1, updated_cloud])
    logger.info(f"Rebuilt PointCloudBatch: {rebuilt_batch}")

    # ======================= Visualize =========================================
    rr.init("point_cloud_batch_example", spawn=True)
    datatypes.visualize(
        point_cloud_batch,
        entity_path="/point_cloud_batch/original",
        label=["Cloud 1", "Cloud 2"],
    )
    datatypes.visualize(
        rebuilt_batch,
        entity_path="/point_cloud_batch/rebuilt",
        label=["Cloud 1", "Updated Cloud 2"],
    )

    # ======================= Serialize / Deserialize ===========================
    start = time.perf_counter()
    serialized = datatypes.serialize(point_cloud_batch)
    serialization_ms = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    deserialized = datatypes.deserialize(serialized)["param_0"]
    deserialization_ms = (time.perf_counter() - start) * 1000

    logger.info(f"Deserialized PointCloudBatch: {deserialized}")
    logger.info(f"Round-trip successful: {point_cloud_batch == deserialized}")
    logger.info(f"Serialization time: {serialization_ms:.3f} ms")
    logger.info(f"Deserialization time: {deserialization_ms:.3f} ms")


if __name__ == "__main__":
    point_cloud_batch_example()
