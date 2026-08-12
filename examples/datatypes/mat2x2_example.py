"""
Example script to demonstrate usage of Mat2x2 datatype.
"""

import time

import numpy as np
from loguru import logger
import rerun as rr

from telekinesis import datatypes


def mat2x2_example():
    """
    Example function to demonstrate usage of Mat2x2 datatype.
        - Create a Mat2x2 data
        - Access the underlying matrix data
        - Visualize the Mat2x2 data using Rerun
        - Update the underlying matrix data
        - Operate on the underlying data with numpy
        - Serialize to PyArrow and back
    """
    # Create a Mat2x2 data can be list or numpy array of shape (2, 2)
    matrix = [[1.0, 2.0], [3.0, 4.0]]
    my_mat2x2 = datatypes.Mat2x2(matrix)
    logger.info(f"Original Mat2x2: {my_mat2x2}")

    # Access the underlying matrix data
    my_mat2x2_data = my_mat2x2.data
    my_mat2x2_shape = my_mat2x2.shape
    my_mat2x2_size = my_mat2x2.size
    my_mat2x2_dtype = my_mat2x2.dtype
    my_mat2x2_ndim = my_mat2x2.ndim
    my_mat2x2_numpy = my_mat2x2.to_numpy()
    my_mat2x2_copy = my_mat2x2.copy()

    logger.info(f"Underlying Mat2x2 data: {my_mat2x2_data}")
    logger.info(f"Underlying Mat2x2 shape: {my_mat2x2_shape}")
    logger.info(f"Underlying Mat2x2 size: {my_mat2x2_size}")
    logger.info(f"Underlying Mat2x2 dtype: {my_mat2x2_dtype}")
    logger.info(f"Underlying Mat2x2 ndim: {my_mat2x2_ndim}")
    logger.info(f"Underlying Mat2x2 numpy array: {my_mat2x2_numpy}")
    logger.info(f"Underlying Mat2x2 object: {my_mat2x2_copy}")

    # Visualize the Mat2x2 data using Rerun
    logger.info("Visualizing with Rerun...")
    rr.init("mat2x2_example", spawn=True)
    datatypes.visualize(my_mat2x2, entity_path="/Mat2x2")

    # Update the my_mat2x2_data
    new_mat2x2_data = [[5.0, 6.0], [7.0, 8.0]]
    my_mat2x2.data = new_mat2x2_data
    logger.info(f"Updated Mat2x2: {my_mat2x2}")
    datatypes.visualize(my_mat2x2, entity_path="/Mat2x2/updated")

    # Operate on the underlying data with numpy - Add to the matrix
    my_mat2x2_sum = my_mat2x2 + np.array([[1.0, 1.0], [1.0, 1.0]])
    logger.info(f"Sum of Mat2x2 with numpy array: {my_mat2x2_sum}")

    # Serialize to PyArrow and back
    serialization_start_time = time.perf_counter()
    serialized = datatypes.serialize(my_mat2x2)
    serialization_end_time = time.perf_counter()

    deserialization_start_time = time.perf_counter()
    deserialized = datatypes.deserialize(serialized)["param_0"]
    deserialization_end_time = time.perf_counter()
    logger.info(f"Deserialized Mat2x2: {deserialized}")
    logger.info(f"Deserialized Mat2x2 data: {deserialized.data == new_mat2x2_data}")

    logger.info(
        f"Serialized Mat2x2 to PyArrow in {(serialization_end_time - serialization_start_time) * 1000:.6f} ms."
    )
    logger.info(
        f"Deserialized Mat2x2 from PyArrow in {(deserialization_end_time - deserialization_start_time) * 1000:.6f} ms."
    )


if __name__ == "__main__":
    mat2x2_example()
