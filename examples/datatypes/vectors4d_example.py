"""
Example script to demonstrate usage of Vectors4D datatype.
"""

import time

import numpy as np
from loguru import logger
import rerun as rr

from telekinesis import datatypes


def vectors4d_example():
    """
    Example function to demonstrate usage of Vectors4D datatype.
        - Create a Vectors4D data
        - Access the underlying vectors data
        - Visualize the Vectors4D data using Rerun
        - Update the underlying vectors data
        - Operate on the underlying data with numpy
        - Serialize to PyArrow and back
        - Construct an empty (zero-row) batch
    """
    # Create a Vectors4D data can be list or numpy array of shape (N, 4)
    vectors = [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]
    my_vectors4d = datatypes.Vectors4D(vectors)
    logger.info(f"Original Vectors4D: {my_vectors4d}")

    # Access the underlying vectors data
    my_vectors4d_data = my_vectors4d.data
    my_vectors4d_shape = my_vectors4d.shape
    my_vectors4d_size = my_vectors4d.size
    my_vectors4d_dtype = my_vectors4d.dtype
    my_vectors4d_ndim = my_vectors4d.ndim
    my_vectors4d_numpy = my_vectors4d.to_numpy()
    my_vectors4d_copy = my_vectors4d.copy()

    logger.info(f"Underlying Vectors4D data: {my_vectors4d_data}")
    logger.info(f"Underlying Vectors4D shape: {my_vectors4d_shape}")
    logger.info(f"Underlying Vectors4D size: {my_vectors4d_size}")
    logger.info(f"Underlying Vectors4D dtype: {my_vectors4d_dtype}")
    logger.info(f"Underlying Vectors4D ndim: {my_vectors4d_ndim}")
    logger.info(f"Underlying Vectors4D numpy array: {my_vectors4d_numpy}")
    logger.info(f"Underlying Vectors4D object: {my_vectors4d_copy}")

    # Visualize the Vectors4D data using Rerun
    logger.info("Visualizing with Rerun...")
    rr.init("vectors4d_example", spawn=True)
    datatypes.visualize(my_vectors4d, entity_path="/Vectors4D")

    # Update the my_vectors4d_data
    new_vectors4d_data = [[9.0, 10.0, 11.0, 12.0], [13.0, 14.0, 15.0, 16.0]]
    my_vectors4d.data = new_vectors4d_data
    logger.info(f"Updated Vectors4D: {my_vectors4d}")
    datatypes.visualize(my_vectors4d, entity_path="/Vectors4D/updated")

    # Operate on the underlying data with numpy - Add to the vectors
    my_vectors4d_sum = my_vectors4d + np.array([1.0, 1.0, 1.0, 1.0])
    logger.info(f"Sum of Vectors4D with numpy array: {my_vectors4d_sum}")

    # Serialize to PyArrow and back
    serialization_start_time = time.perf_counter()
    serialized = datatypes.serialize(my_vectors4d)
    serialization_end_time = time.perf_counter()

    deserialization_start_time = time.perf_counter()
    deserialized = datatypes.deserialize(serialized)["param_0"]
    deserialization_end_time = time.perf_counter()
    logger.info(f"Deserialized Vectors4D: {deserialized}")
    logger.info(f"Deserialized Vectors4D data: {deserialized.data == new_vectors4d_data}")

    logger.info(
        f"Serialized Vectors4D to PyArrow in {(serialization_end_time - serialization_start_time) * 1000:.6f} ms."
    )
    logger.info(
        f"Deserialized Vectors4D from PyArrow in {(deserialization_end_time - deserialization_start_time) * 1000:.6f} ms."
    )

    # Vectors4D also accepts an empty (zero-row) batch, since shape_spec's leading
    # axis (None) allows N=0. A plain `[]` won't work since it has no second axis,
    # so an explicitly-shaped empty array is required.
    empty_vectors4d = datatypes.Vectors4D(np.empty((0, 4), dtype=np.float32))
    logger.info(f"Empty Vectors4D: {empty_vectors4d}")
    logger.info(f"Empty Vectors4D shape: {empty_vectors4d.shape}")


if __name__ == "__main__":
    vectors4d_example()
