"""
Example script to demonstrate usage of Vector4D datatype.
"""

import time

import numpy as np
from loguru import logger
import rerun as rr

from telekinesis import datatypes


def vector4d_example():
    """
    Example function to demonstrate usage of Vector4D datatype.
        - Create a Vector4D data
        - Access the underlying vector data
        - Visualize the Vector4D data using Rerun
        - Update the underlying vector data
        - Operate on the underlying data with numpy
        - Serialize to PyArrow and back
    """
    # Create a Vector4D data can be list or numpy array of shape (4,)
    vector = [1.0, 2.0, 3.0, 4.0]
    my_vector4d = datatypes.Vector4D(vector)
    logger.info(f"Original Vector4D: {my_vector4d}")

    # Access the underlying vector data
    my_vector4d_data = my_vector4d.data
    my_vector4d_shape = my_vector4d.shape
    my_vector4d_size = my_vector4d.size
    my_vector4d_dtype = my_vector4d.dtype
    my_vector4d_ndim = my_vector4d.ndim
    my_vector4d_numpy = my_vector4d.to_numpy()
    my_vector4d_copy = my_vector4d.copy()

    logger.info(f"Underlying Vector4D data: {my_vector4d_data}")
    logger.info(f"Underlying Vector4D shape: {my_vector4d_shape}")
    logger.info(f"Underlying Vector4D size: {my_vector4d_size}")
    logger.info(f"Underlying Vector4D dtype: {my_vector4d_dtype}")
    logger.info(f"Underlying Vector4D ndim: {my_vector4d_ndim}")
    logger.info(f"Underlying Vector4D numpy array: {my_vector4d_numpy}")
    logger.info(f"Underlying Vector4D object: {my_vector4d_copy}")

    # Visualize the Vector4D data using Rerun
    logger.info("Visualizing with Rerun...")
    rr.init("vector4d_example", spawn=True)
    datatypes.visualize(my_vector4d, entity_path="/Vector4D")

    # Update the my_vector4d_data
    new_vector4d_data = [5.0, 6.0, 7.0, 8.0]
    my_vector4d.data = new_vector4d_data
    logger.info(f"Updated Vector4D: {my_vector4d}")
    datatypes.visualize(my_vector4d, entity_path="/Vector4D/updated")

    # Operate on the underlying data with numpy - Add to the vector
    my_vector4d_sum = my_vector4d + np.array([1.0, 1.0, 1.0, 1.0])
    logger.info(f"Sum of Vector4D with numpy array: {my_vector4d_sum}")

    # Serialize to PyArrow and back
    serialization_start_time = time.perf_counter()
    serialized = datatypes.serialize(my_vector4d)
    serialization_end_time = time.perf_counter()

    deserialization_start_time = time.perf_counter()
    deserialized = datatypes.deserialize(serialized)["param_0"]
    deserialization_end_time = time.perf_counter()
    logger.info(f"Deserialized Vector4D: {deserialized}")
    logger.info(f"Deserialized Vector4D data: {deserialized.data == new_vector4d_data}")

    logger.info(
        f"Serialized Vector4D to PyArrow in {(serialization_end_time - serialization_start_time) * 1000:.6f} ms."
    )
    logger.info(
        f"Deserialized Vector4D from PyArrow in {(deserialization_end_time - deserialization_start_time) * 1000:.6f} ms."
    )


if __name__ == "__main__":
    vector4d_example()
