"""
Example script to demonstrate usage of Vectors3D datatype.
"""

import time

import numpy as np
from loguru import logger
import rerun as rr

from telekinesis import datatypes


def vectors3d_example():
    """
    Example function to demonstrate usage of Vectors3D datatype.
        - Create a Vectors3D data
        - Access the underlying vectors data
        - Visualize the Vectors3D data using Rerun
        - Update the underlying vectors data
        - Operate on the underlying data with numpy
        - Serialize to PyArrow and back
        - Construct an empty (zero-row) batch
    """
    # Create a Vectors3D data can be list or numpy array of shape (N, 3)
    vectors = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    my_vectors3d = datatypes.Vectors3D(vectors)
    logger.info(f"Original Vectors3D: {my_vectors3d}")

    # Access the underlying vectors data
    my_vectors3d_data = my_vectors3d.data
    my_vectors3d_shape = my_vectors3d.shape
    my_vectors3d_size = my_vectors3d.size
    my_vectors3d_dtype = my_vectors3d.dtype
    my_vectors3d_ndim = my_vectors3d.ndim
    my_vectors3d_numpy = my_vectors3d.to_numpy()
    my_vectors3d_copy = my_vectors3d.copy()

    logger.info(f"Underlying Vectors3D data: {my_vectors3d_data}")
    logger.info(f"Underlying Vectors3D shape: {my_vectors3d_shape}")
    logger.info(f"Underlying Vectors3D size: {my_vectors3d_size}")
    logger.info(f"Underlying Vectors3D dtype: {my_vectors3d_dtype}")
    logger.info(f"Underlying Vectors3D ndim: {my_vectors3d_ndim}")
    logger.info(f"Underlying Vectors3D numpy array: {my_vectors3d_numpy}")
    logger.info(f"Underlying Vectors3D object: {my_vectors3d_copy}")

    # Visualize the Vectors3D data using Rerun
    logger.info("Visualizing with Rerun...")
    rr.init("vectors3d_example", spawn=True)
    datatypes.visualize(my_vectors3d, entity_path="/Vectors3D", label=["Vector 1", "Vector 2"])

    # Update the my_vectors3d_data
    new_vectors3d_data = [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]
    my_vectors3d.data = new_vectors3d_data
    logger.info(f"Updated Vectors3D: {my_vectors3d}")
    datatypes.visualize(
        my_vectors3d,
        entity_path="/Vectors3D/updated",
        label=["Updated Vector 1", "Updated Vector 2"],
    )

    # Operate on the underlying data with numpy - Add to the vectors
    my_vectors3d_sum = my_vectors3d + np.array([1.0, 1.0, 1.0])
    logger.info(f"Sum of Vectors3D with numpy array: {my_vectors3d_sum}")

    # Serialize to PyArrow and back
    serialization_start_time = time.perf_counter()
    serialized = datatypes.serialize(my_vectors3d)
    serialization_end_time = time.perf_counter()

    deserialization_start_time = time.perf_counter()
    deserialized = datatypes.deserialize(serialized)["param_0"]
    deserialization_end_time = time.perf_counter()
    logger.info(f"Deserialized Vectors3D: {deserialized}")
    logger.info(f"Deserialized Vectors3D data: {deserialized.data == new_vectors3d_data}")

    logger.info(
        f"Serialized Vectors3D to PyArrow in {(serialization_end_time - serialization_start_time) * 1000:.6f} ms."
    )
    logger.info(
        f"Deserialized Vectors3D from PyArrow in {(deserialization_end_time - deserialization_start_time) * 1000:.6f} ms."
    )

    # Vectors3D also accepts an empty (zero-row) batch, since shape_spec's leading
    # axis (None) allows N=0. A plain `[]` won't work since it has no second axis,
    # so an explicitly-shaped empty array is required.
    empty_vectors3d = datatypes.Vectors3D(np.empty((0, 3), dtype=np.float32))
    logger.info(f"Empty Vectors3D: {empty_vectors3d}")
    logger.info(f"Empty Vectors3D shape: {empty_vectors3d.shape}")


if __name__ == "__main__":
    vectors3d_example()
