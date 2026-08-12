"""
Example script to demonstrate usage of Mat4x4 datatype.
"""

import time

import numpy as np
from loguru import logger
import rerun as rr

from telekinesis import datatypes


def mat4x4_example():
    """
    Example function to demonstrate usage of Mat4x4 datatype.
        - Create a Mat4x4 data
        - Access the underlying matrix data
        - Visualize the Mat4x4 data using Rerun
        - Update the underlying matrix data
        - Operate on the underlying data with numpy
        - Serialize to PyArrow and back
    """
    # Create a Mat4x4 data can be list or numpy array of shape (4, 4)
    matrix = [
        [1.0, 2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0, 8.0],
        [9.0, 10.0, 11.0, 12.0],
        [13.0, 14.0, 15.0, 16.0],
    ]
    my_mat4x4 = datatypes.Mat4x4(matrix)
    logger.info(f"Original Mat4x4: {my_mat4x4}")

    # Access the underlying matrix data
    my_mat4x4_data = my_mat4x4.data
    my_mat4x4_shape = my_mat4x4.shape
    my_mat4x4_size = my_mat4x4.size
    my_mat4x4_dtype = my_mat4x4.dtype
    my_mat4x4_ndim = my_mat4x4.ndim
    my_mat4x4_numpy = my_mat4x4.to_numpy()
    my_mat4x4_copy = my_mat4x4.copy()

    logger.info(f"Underlying Mat4x4 data: {my_mat4x4_data}")
    logger.info(f"Underlying Mat4x4 shape: {my_mat4x4_shape}")
    logger.info(f"Underlying Mat4x4 size: {my_mat4x4_size}")
    logger.info(f"Underlying Mat4x4 dtype: {my_mat4x4_dtype}")
    logger.info(f"Underlying Mat4x4 ndim: {my_mat4x4_ndim}")
    logger.info(f"Underlying Mat4x4 numpy array: {my_mat4x4_numpy}")
    logger.info(f"Underlying Mat4x4 object: {my_mat4x4_copy}")

    # Visualize the Mat4x4 data using Rerun
    logger.info("Visualizing with Rerun...")
    rr.init("mat4x4_example", spawn=True)
    datatypes.visualize(my_mat4x4, entity_path="/Mat4x4")

    # Update the my_mat4x4_data
    new_mat4x4_data = [
        [16.0, 15.0, 14.0, 13.0],
        [12.0, 11.0, 10.0, 9.0],
        [8.0, 7.0, 6.0, 5.0],
        [4.0, 3.0, 2.0, 1.0],
    ]
    my_mat4x4.data = new_mat4x4_data
    logger.info(f"Updated Mat4x4: {my_mat4x4}")
    datatypes.visualize(my_mat4x4, entity_path="/Mat4x4/updated")

    # Operate on the underlying data with numpy - Add to the matrix
    my_mat4x4_sum = my_mat4x4 + np.array(
        [
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
        ]
    )
    logger.info(f"Sum of Mat4x4 with numpy array: {my_mat4x4_sum}")

    # Serialize to PyArrow and back
    serialization_start_time = time.perf_counter()
    serialized = datatypes.serialize(my_mat4x4)
    serialization_end_time = time.perf_counter()

    deserialization_start_time = time.perf_counter()
    deserialized = datatypes.deserialize(serialized)["param_0"]
    deserialization_end_time = time.perf_counter()
    logger.info(f"Deserialized Mat4x4: {deserialized}")
    logger.info(f"Deserialized Mat4x4 data: {deserialized.data == new_mat4x4_data}")

    logger.info(
        f"Serialized Mat4x4 to PyArrow in {(serialization_end_time - serialization_start_time) * 1000:.6f} ms."
    )
    logger.info(
        f"Deserialized Mat4x4 from PyArrow in {(deserialization_end_time - deserialization_start_time) * 1000:.6f} ms."
    )


if __name__ == "__main__":
    mat4x4_example()
