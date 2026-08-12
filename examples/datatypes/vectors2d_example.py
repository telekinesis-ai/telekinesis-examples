"""
Example script to demonstrate usage of Vectors2D datatype.
"""

import time

import numpy as np
from loguru import logger
import rerun as rr

from telekinesis import datatypes


def vectors2d_example():
    """
    Example function to demonstrate usage of Vectors2D datatype.
        - Create a Vectors2D data
        - Access the underlying vectors data
        - Visualize the Vectors2D data using Rerun
        - Update the underlying vectors data
        - Operate on the underlying data with numpy
        - Serialize to PyArrow and back
        - Construct an empty (zero-row) batch
    """
    # Create a Vectors2D data can be list or numpy array of shape (N, 2)
    vectors = [[1.0, 2.0], [3.0, 4.0]]
    my_vectors2d = datatypes.Vectors2D(vectors)
    logger.info(f"Original Vectors2D: {my_vectors2d}")

    # Access the underlying vectors data
    my_vectors2d_data = my_vectors2d.data
    my_vectors2d_shape = my_vectors2d.shape
    my_vectors2d_size = my_vectors2d.size
    my_vectors2d_dtype = my_vectors2d.dtype
    my_vectors2d_ndim = my_vectors2d.ndim
    my_vectors2d_numpy = my_vectors2d.to_numpy()
    my_vectors2d_copy = my_vectors2d.copy()

    logger.info(f"Underlying Vectors2D data: {my_vectors2d_data}")
    logger.info(f"Underlying Vectors2D shape: {my_vectors2d_shape}")
    logger.info(f"Underlying Vectors2D size: {my_vectors2d_size}")
    logger.info(f"Underlying Vectors2D dtype: {my_vectors2d_dtype}")
    logger.info(f"Underlying Vectors2D ndim: {my_vectors2d_ndim}")
    logger.info(f"Underlying Vectors2D numpy array: {my_vectors2d_numpy}")
    logger.info(f"Underlying Vectors2D object: {my_vectors2d_copy}")

    # Visualize the Vectors2D data using Rerun
    logger.info("Visualizing with Rerun...")
    rr.init("vectors2d_example", spawn=True)
    datatypes.visualize(my_vectors2d, entity_path="/Vectors2D", label=["Vector 1", "Vector 2"])

    # Update the my_vectors2d_data
    new_vectors2d_data = [[5.0, 6.0], [7.0, 8.0]]
    my_vectors2d.data = new_vectors2d_data
    logger.info(f"Updated Vectors2D: {my_vectors2d}")
    datatypes.visualize(
        my_vectors2d,
        entity_path="/Vectors2D/updated",
        label=["Updated Vector 1", "Updated Vector 2"],
    )

    # Operate on the underlying data with numpy - Add to the vectors
    my_vectors2d_sum = my_vectors2d + np.array([1.0, 1.0])
    logger.info(f"Sum of Vectors2D with numpy array: {my_vectors2d_sum}")

    # Serialize to PyArrow and back
    serialization_start_time = time.perf_counter()
    serialized = datatypes.serialize(my_vectors2d)
    serialization_end_time = time.perf_counter()

    deserialization_start_time = time.perf_counter()
    deserialized = datatypes.deserialize(serialized)["param_0"]
    deserialization_end_time = time.perf_counter()
    logger.info(f"Deserialized Vectors2D: {deserialized}")
    logger.info(f"Deserialized Vectors2D data: {deserialized.data == new_vectors2d_data}")

    logger.info(
        f"Serialized Vectors2D to PyArrow in {(serialization_end_time - serialization_start_time) * 1000:.6f} ms."
    )
    logger.info(
        f"Deserialized Vectors2D from PyArrow in {(deserialization_end_time - deserialization_start_time) * 1000:.6f} ms."
    )

    # Vectors2D also accepts an empty (zero-row) batch, since shape_spec's leading
    # axis (None) allows N=0. A plain `[]` won't work since it has no second axis,
    # so an explicitly-shaped empty array is required.
    empty_vectors2d = datatypes.Vectors2D(np.empty((0, 2), dtype=np.float32))
    logger.info(f"Empty Vectors2D: {empty_vectors2d}")
    logger.info(f"Empty Vectors2D shape: {empty_vectors2d.shape}")


if __name__ == "__main__":
    vectors2d_example()
