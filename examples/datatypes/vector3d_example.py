"""
Example script to demonstrate usage of Vector3D datatype.
"""

import time

import numpy as np
from loguru import logger
import rerun as rr

from telekinesis import datatypes


def vector3d_example():
    """
    Example function to demonstrate usage of Vector3D datatype.
        - Create a Vector3D data
        - Access the underlying vector data
        - Visualize the Vector3D data using Rerun
        - Update the underlying vector data
        - Operate on the underlying data with numpy
        - Serialize to PyArrow and back
    """
    # Create a Vector3D data can be list or numpy array of shape (3,)
    vector = [1.0, 2.0, 3.0]
    my_vector3d = datatypes.Vector3D(vector)
    logger.info(f"Original Vector3D: {my_vector3d}")

    # Access the underlying vector data
    my_vector3d_data = my_vector3d.data
    my_vector3d_shape = my_vector3d.shape
    my_vector3d_size = my_vector3d.size
    my_vector3d_dtype = my_vector3d.dtype
    my_vector3d_ndim = my_vector3d.ndim
    my_vector3d_numpy = my_vector3d.to_numpy()
    my_vector3d_copy = my_vector3d.copy()

    logger.info(f"Underlying Vector3D data: {my_vector3d_data}")
    logger.info(f"Underlying Vector3D shape: {my_vector3d_shape}")
    logger.info(f"Underlying Vector3D size: {my_vector3d_size}")
    logger.info(f"Underlying Vector3D dtype: {my_vector3d_dtype}")
    logger.info(f"Underlying Vector3D ndim: {my_vector3d_ndim}")
    logger.info(f"Underlying Vector3D numpy array: {my_vector3d_numpy}")
    logger.info(f"Underlying Vector3D object: {my_vector3d_copy}")

    # Visualize the Vector3D data using Rerun
    logger.info("Visualizing with Rerun...")
    rr.init("vector3d_example", spawn=True)
    datatypes.visualize(my_vector3d, entity_path="/Vector3D", label="My Vector3D")

    # Update the my_vector3d_data
    new_vector3d_data = [4.0, 5.0, 6.0]
    my_vector3d.data = new_vector3d_data
    logger.info(f"Updated Vector3D: {my_vector3d}")
    datatypes.visualize(my_vector3d, entity_path="/Vector3D/updated", label="Updated Vector3D")

    # Operate on the underlying data with numpy - Add to the vector
    my_vector3d_sum = my_vector3d + np.array([1.0, 1.0, 1.0])
    logger.info(f"Sum of Vector3D with numpy array: {my_vector3d_sum}")

    # Serialize to PyArrow and back
    serialization_start_time = time.perf_counter()
    serialized = datatypes.serialize(my_vector3d)
    serialization_end_time = time.perf_counter()

    deserialization_start_time = time.perf_counter()
    deserialized = datatypes.deserialize(serialized)["param_0"]
    deserialization_end_time = time.perf_counter()
    logger.info(f"Deserialized Vector3D: {deserialized}")
    logger.info(f"Deserialized Vector3D data: {deserialized.data == new_vector3d_data}")

    logger.info(
        f"Serialized Vector3D to PyArrow in {(serialization_end_time - serialization_start_time) * 1000:.6f} ms."
    )
    logger.info(
        f"Deserialized Vector3D from PyArrow in {(deserialization_end_time - deserialization_start_time) * 1000:.6f} ms."
    )


if __name__ == "__main__":
    vector3d_example()
