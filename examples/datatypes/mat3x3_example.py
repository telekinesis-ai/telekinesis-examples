"""
Example script to demonstrate usage of Mat3x3 datatype.
"""

import time

import numpy as np
from loguru import logger
import rerun as rr

from telekinesis import datatypes


def mat3x3_example():
    """
    Example function to demonstrate usage of Mat3x3 datatype.
        - Create a Mat3x3 data
        - Access the underlying matrix data
        - Visualize the Mat3x3 data using Rerun
        - Update the underlying matrix data
        - Operate on the underlying data with numpy
        - Serialize to PyArrow and back
    """
    # Create a Mat3x3 data can be list or numpy array of shape (3, 3)
    matrix = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
    my_mat3x3 = datatypes.Mat3x3(matrix)
    logger.info(f"Original Mat3x3: {my_mat3x3}")

    # Access the underlying matrix data
    my_mat3x3_data = my_mat3x3.data
    my_mat3x3_shape = my_mat3x3.shape
    my_mat3x3_size = my_mat3x3.size
    my_mat3x3_dtype = my_mat3x3.dtype
    my_mat3x3_ndim = my_mat3x3.ndim
    my_mat3x3_numpy = my_mat3x3.to_numpy()
    my_mat3x3_copy = my_mat3x3.copy()

    logger.info(f"Underlying Mat3x3 data: {my_mat3x3_data}")
    logger.info(f"Underlying Mat3x3 shape: {my_mat3x3_shape}")
    logger.info(f"Underlying Mat3x3 size: {my_mat3x3_size}")
    logger.info(f"Underlying Mat3x3 dtype: {my_mat3x3_dtype}")
    logger.info(f"Underlying Mat3x3 ndim: {my_mat3x3_ndim}")
    logger.info(f"Underlying Mat3x3 numpy array: {my_mat3x3_numpy}")
    logger.info(f"Underlying Mat3x3 object: {my_mat3x3_copy}")

    # Visualize the Mat3x3 data using Rerun
    logger.info("Visualizing with Rerun...")
    rr.init("mat3x3_example", spawn=True)
    datatypes.visualize(my_mat3x3, entity_path="/Mat3x3")

    # Update the my_mat3x3_data
    new_mat3x3_data = [[9.0, 8.0, 7.0], [6.0, 5.0, 4.0], [3.0, 2.0, 1.0]]
    my_mat3x3.data = new_mat3x3_data
    logger.info(f"Updated Mat3x3: {my_mat3x3}")
    datatypes.visualize(my_mat3x3, entity_path="/Mat3x3/updated")

    # Operate on the underlying data with numpy - Add to the matrix
    my_mat3x3_sum = my_mat3x3 + np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
    logger.info(f"Sum of Mat3x3 with numpy array: {my_mat3x3_sum}")

    # Serialize to PyArrow and back
    serialization_start_time = time.perf_counter()
    serialized = datatypes.serialize(my_mat3x3)
    serialization_end_time = time.perf_counter()

    deserialization_start_time = time.perf_counter()
    deserialized = datatypes.deserialize(serialized)["param_0"]
    deserialization_end_time = time.perf_counter()
    logger.info(f"Deserialized Mat3x3: {deserialized}")
    logger.info(f"Deserialized Mat3x3 data: {deserialized.data == new_mat3x3_data}")

    logger.info(
        f"Serialized Mat3x3 to PyArrow in {(serialization_end_time - serialization_start_time) * 1000:.6f} ms."
    )
    logger.info(
        f"Deserialized Mat3x3 from PyArrow in {(deserialization_end_time - deserialization_start_time) * 1000:.6f} ms."
    )


if __name__ == "__main__":
    mat3x3_example()
