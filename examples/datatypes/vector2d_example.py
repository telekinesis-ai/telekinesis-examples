"""
Example script to demonstrate usage of Vector2D datatype.
"""

import time

import numpy as np
from loguru import logger
import rerun as rr

from telekinesis import datatypes


def vector2d_example():
    """
    Example function to demonstrate usage of Vector2D datatype.
        - Create a Vector2D data
        - Access the underlying vector data
        - Visualize the Vector2D data using Rerun
        - Update the underlying vector data
        - Operate on the underlying data with numpy
        - Serialize to PyArrow and back
    """
    # Create a Vector2D data can be list or numpy array of shape (2,)
    vector = [1.0, 2.0]
    my_vector2d = datatypes.Vector2D(vector)
    logger.info(f"Original Vector2D: {my_vector2d}")

    # Access the underlying vector data
    my_vector2d_data = my_vector2d.data
    my_vector2d_shape = my_vector2d.shape
    my_vector2d_size = my_vector2d.size
    my_vector2d_dtype = my_vector2d.dtype
    my_vector2d_ndim = my_vector2d.ndim
    my_vector2d_numpy = my_vector2d.to_numpy()
    my_vector2d_copy = my_vector2d.copy()

    logger.info(f"Underlying Vector2D data: {my_vector2d_data}")
    logger.info(f"Underlying Vector2D shape: {my_vector2d_shape}")
    logger.info(f"Underlying Vector2D size: {my_vector2d_size}")
    logger.info(f"Underlying Vector2D dtype: {my_vector2d_dtype}")
    logger.info(f"Underlying Vector2D ndim: {my_vector2d_ndim}")
    logger.info(f"Underlying Vector2D numpy array: {my_vector2d_numpy}")
    logger.info(f"Underlying Vector2D object: {my_vector2d_copy}")

    # Visualize the Vector2D data using Rerun
    logger.info("Visualizing with Rerun...")
    rr.init("vector2d_example", spawn=True)
    datatypes.visualize(my_vector2d, entity_path="/Vector2D", label="My Vector2D")

    # Update the my_vector2d_data
    new_vector2d_data = [3.0, 4.0]
    my_vector2d.data = new_vector2d_data
    logger.info(f"Updated Vector2D: {my_vector2d}")
    datatypes.visualize(my_vector2d, entity_path="/Vector2D/updated", label="Updated Vector2D")

    # Operate on the underlying data with numpy - Add to the vector
    my_vector2d_sum = my_vector2d + np.array([1.0, 1.0])
    logger.info(f"Sum of Vector2D with numpy array: {my_vector2d_sum}")

    # Serialize to PyArrow and back
    serialization_start_time = time.perf_counter()
    serialized = datatypes.serialize(my_vector2d)
    serialization_end_time = time.perf_counter()

    deserialization_start_time = time.perf_counter()
    deserialized = datatypes.deserialize(serialized)["param_0"]
    deserialization_end_time = time.perf_counter()
    logger.info(f"Deserialized Vector2D: {deserialized}")
    logger.info(f"Deserialized Vector2D data: {deserialized.data == new_vector2d_data}")

    logger.info(
        f"Serialized Vector2D to PyArrow in {(serialization_end_time - serialization_start_time) * 1000:.6f} ms."
    )
    logger.info(
        f"Deserialized Vector2D from PyArrow in {(deserialization_end_time - deserialization_start_time) * 1000:.6f} ms."
    )


if __name__ == "__main__":
    vector2d_example()
