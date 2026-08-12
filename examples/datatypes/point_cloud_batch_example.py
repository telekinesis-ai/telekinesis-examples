"""
Example script to demonstrate usage of PointCloudBatch datatype.
"""

import time

import numpy as np
from loguru import logger
import rerun as rr

from telekinesis import datatypes


def point_cloud_batch_example():
    """
    Example function to demonstrate usage of PointCloudBatch datatype.
     - Create a PointCloudBatch data
     - Print the original data
    """
    # Create a PointCloudBatch data
    N = 4000000
    # Each cloud gets its own position range (offset along x) so they render
    # as separate, non-overlapping clusters instead of piling up at the origin.
    input_point_cloud_1 = datatypes.PointCloud(
        np.random.randn(N, 3).astype(np.float32) + np.array([-10.0, 0.0, 0.0], dtype=np.float32),
        normals=np.random.randn(N, 3).astype(np.float32),
        colors=np.random.randint(0, 255, (N, 3), dtype=np.uint8),
        use_compression=False,
    )
    # Bare ndarray: positions only
    input_point_cloud_2 = np.random.randn(N, 3).astype(np.float32) + np.array(
        [10.0, 0.0, 0.0], dtype=np.float32
    )
    input_point_cloud_3 = np.random.randn(N, 3).astype(np.float32) + np.array(
        [10.0, 0.0, 0.0], dtype=np.float32
    )
    point_clouds = [
        input_point_cloud_1,
        input_point_cloud_2,
        input_point_cloud_3,
    ]
    my_point_cloud_batch = datatypes.PointCloudBatch(point_clouds)
    logger.info(f"Original PointCloudBatch: {my_point_cloud_batch}")

    logger.info("Visualizing with Rerun...")
    rr.init("point_cloud_batch_example", spawn=True)
    datatypes.visualize(
        my_point_cloud_batch,
        entity_path="/PointCloudBatch/my_point_cloud_batch",
        label=["My PointCloud 1", "My PointCloud 2", "My PointCloud 3"],
    )

    # Access underlying positions / normals / colors
    my_point_cloud_batch_positions = my_point_cloud_batch.positions
    my_point_cloud_batch_normals = my_point_cloud_batch.normals
    my_point_cloud_batch_colors = my_point_cloud_batch.colors
    my_point_cloud_batch_length = len(my_point_cloud_batch)
    logger.info(f"Underlying PointCloudBatch positions: {my_point_cloud_batch_positions}")
    logger.info(f"Underlying PointCloudBatch normals: {my_point_cloud_batch_normals}")
    logger.info(f"Underlying PointCloudBatch colors: {my_point_cloud_batch_colors}")
    logger.info(f"Length of PointCloudBatch: {my_point_cloud_batch_length}")

    # Index the PointCloudBatch to get a single PointCloud object
    index = 0
    my_single_point_cloud = my_point_cloud_batch[index]
    logger.info(f"Single PointCloud from batch: {my_single_point_cloud}")
    datatypes.visualize(
        my_single_point_cloud,
        entity_path="/PointCloudBatch/my_updated_point_cloud_1",
        label="My Updated PointCloud 1",
    )

    # PointCloudBatch is immutable after construction -- there is no way to
    # replace a cloud in place. To change a cloud, build a new PointCloudBatch
    # with the updated PointCloud in place of the one you want to replace.
    index = -1
    updated_point_cloud = datatypes.PointCloud(
        np.random.randn(N, 3).astype(np.float32),
        normals=np.random.randn(N, 3).astype(np.float32),
        colors=np.full((N, 3), [255, 0, 0], dtype=np.uint8),  # e.g. solid red for all N points
        use_compression=True,
    )
    datatypes.visualize(
        updated_point_cloud,
        entity_path="/PointCloudBatch/my_updated_point_cloud_2",
        label="My Updated PointCloud 2",
    )
    rebuild_start_time = time.perf_counter()
    point_clouds[index] = updated_point_cloud
    my_point_cloud_batch = datatypes.PointCloudBatch(point_clouds)
    rebuild_end_time = time.perf_counter()
    logger.info(f"Rebuilt PointCloudBatch: {my_point_cloud_batch}")
    logger.info(f"Rebuild time: {(rebuild_end_time - rebuild_start_time) * 1000:.6f} ms")

    # Serialize to PyArrow and back
    serialization_start_time = time.perf_counter()
    serialized_point_cloud_batch = datatypes.serialize(my_point_cloud_batch)
    serialization_end_time = time.perf_counter()
    logger.info("Serialized PointCloudBatch")

    deserialization_start_time = time.perf_counter()
    deserialized_point_cloud_batch = datatypes.deserialize(serialized_point_cloud_batch)["param_0"]
    deserialization_end_time = time.perf_counter()

    logger.info(f"Deserialized PointCloudBatch: {deserialized_point_cloud_batch}")
    logger.info(
        f"Deserialized PointCloudBatch matches original: {deserialized_point_cloud_batch == my_point_cloud_batch}"
    )

    logger.info(
        f"Serialization time: {(serialization_end_time - serialization_start_time) * 1000:.6f} ms"
    )
    logger.info(
        f"Deserialization time: {(deserialization_end_time - deserialization_start_time) * 1000:.6f} ms"
    )


if __name__ == "__main__":
    point_cloud_batch_example()
