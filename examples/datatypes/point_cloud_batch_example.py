"""Demonstrates the Telekinesis PointCloudBatch datatype."""

import time

import numpy as np
from loguru import logger
import rerun as rr

from telekinesis import datatypes

def point_cloud_batch_example():
    """Demonstrate creation, visualization, access, indexing, rebuilding, and serialization of a PointCloudBatch."""

    # ======================= Create ============================================
    N = 4000000
    cloud_1 = datatypes.PointCloud(
        np.random.randn(N, 3).astype(np.float32) + np.array([-10.0, 0.0, 0.0], dtype=np.float32),
        normals=np.random.randn(N, 3).astype(np.float32),
        colors=np.random.randint(0, 255, (N, 3), dtype=np.uint8),
        use_compression=False,
    )
    cloud_2 = np.random.randn(N, 3).astype(np.float32) + np.array(
        [10.0, 0.0, 0.0], dtype=np.float32
    )
    cloud_3 = np.random.randn(N, 3).astype(np.float32) + np.array(
        [10.0, 0.0, 0.0], dtype=np.float32
    )
    clouds = [cloud_1, cloud_2, cloud_3]
    batch = datatypes.PointCloudBatch(clouds)

    logger.info(f"Original PointCloudBatch: {batch}")

    # ======================= Visualize =========================================
    rr.init("point_cloud_batch_example", spawn=True)
    datatypes.visualize(
        batch,
        entity_path="/PointCloudBatch/my_point_cloud_batch",
        label=["My PointCloud 1", "My PointCloud 2", "My PointCloud 3"],
    )

    # ======================= Inspect ===========================================
    positions = batch.positions
    normals = batch.normals
    colors = batch.colors
    length = len(batch)

    logger.info(f"length={length}")
    logger.info(f"Underlying positions: {positions}")
    logger.info(f"Underlying normals: {normals}")
    logger.info(f"Underlying colors: {colors}")

    # ======================= Index =============================================
    index = 0
    single_cloud = batch[index]
    logger.info(f"Single PointCloud from batch: {single_cloud}")
    datatypes.visualize(
        single_cloud,
        entity_path="/PointCloudBatch/my_updated_point_cloud_1",
        label="My Updated PointCloud 1",
    )

    # ======================= Rebuild ===========================================
    index = -1
    updated_cloud = datatypes.PointCloud(
        np.random.randn(N, 3).astype(np.float32),
        normals=np.random.randn(N, 3).astype(np.float32),
        colors=np.full((N, 3), [255, 0, 0], dtype=np.uint8),
        use_compression=True,
    )
    datatypes.visualize(
        updated_cloud,
        entity_path="/PointCloudBatch/my_updated_point_cloud_2",
        label="My Updated PointCloud 2",
    )
    start = time.perf_counter()
    clouds[index] = updated_cloud
    batch = datatypes.PointCloudBatch(clouds)
    rebuild_ms = (time.perf_counter() - start) * 1000
    logger.info(f"Rebuilt PointCloudBatch: {batch}")
    logger.info(f"Rebuild time: {rebuild_ms:.3f} ms")

    # ======================= Serialize / Deserialize ===========================
    start = time.perf_counter()
    serialized = datatypes.serialize(batch)
    serialization_ms = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    deserialized = datatypes.deserialize(serialized)["param_0"]
    deserialization_ms = (time.perf_counter() - start) * 1000

    logger.info(f"Deserialized PointCloudBatch: {deserialized}")
    logger.info(f"Round-trip successful: {deserialized == batch}")
    logger.info(f"Serialization time: {serialization_ms:.3f} ms")
    logger.info(f"Deserialization time: {deserialization_ms:.3f} ms")


if __name__ == "__main__":
    point_cloud_batch_example()
