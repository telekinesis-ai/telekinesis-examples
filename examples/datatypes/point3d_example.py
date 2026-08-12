"""
Example script to demonstrate usage of Point3D datatype.
"""

import time

import numpy as np
from loguru import logger
import rerun as rr

from telekinesis import datatypes


def point3d_example():
    """
    Example function to demonstrate usage of Point3D datatype.
        - Create a Point3D data
        - Access the underlying point data
        - Visualize the Point3D data using Rerun
        - Update the underlying point data
        - Operate on the underlying data with numpy
        - Serialize to PyArrow and back
    """
    # Create a Point3D data can be list or numpy array of shape (3,)
    point = [1.0, 2.0, 3.0]
    my_point3d = datatypes.Point3D(point)
    logger.info(f"Original Point3D: {my_point3d}")

    # Access the underlying point data
    my_point3d_data = my_point3d.data
    my_point3d_shape = my_point3d.shape
    my_point3d_size = my_point3d.size
    my_point3d_dtype = my_point3d.dtype
    my_point3d_ndim = my_point3d.ndim
    my_point3d_numpy = my_point3d.to_numpy()
    my_point3d_copy = my_point3d.copy()

    logger.info(f"Underlying Point3D data: {my_point3d_data}")
    logger.info(f"Underlying Point3D shape: {my_point3d_shape}")
    logger.info(f"Underlying Point3D size: {my_point3d_size}")
    logger.info(f"Underlying Point3D dtype: {my_point3d_dtype}")
    logger.info(f"Underlying Point3D ndim: {my_point3d_ndim}")
    logger.info(f"Underlying Point3D numpy array: {my_point3d_numpy}")
    logger.info(f"Underlying Point3D object: {my_point3d_copy}")

    # Visualize the Point3D data using Rerun
    logger.info("Visualizing with Rerun...")
    rr.init("point3d_example", spawn=True)
    datatypes.visualize(my_point3d, entity_path="/Point3D", label="My Point3D")

    # Update the my_point3d_data
    new_point3d_data = [4.0, 5.0, 6.0]
    my_point3d.data = new_point3d_data
    logger.info(f"Updated Point3D: {my_point3d}")
    datatypes.visualize(my_point3d, entity_path="/Point3D/updated", label="Updated Point3D")

    # Operate on the underlying data with numpy - add, subtract, multiply, divide
    my_point3d_sum = my_point3d + np.array([1.0, 1.0, 1.0])
    my_point3d_diff = my_point3d - np.array([1.0, 1.0, 1.0])
    my_point3d_prod = my_point3d * np.array(2.0)
    my_point3d_quot = my_point3d / np.array(2.0)
    logger.info(f"Sum of Point3D with numpy array: {my_point3d_sum}")
    logger.info(f"Difference of Point3D with numpy array: {my_point3d_diff}")
    logger.info(f"Product of Point3D with scalar: {my_point3d_prod}")
    logger.info(f"Quotient of Point3D with scalar: {my_point3d_quot}")

    # Serialize to PyArrow and back
    serialization_start_time = time.perf_counter()
    serialized = datatypes.serialize(my_point3d)
    serialization_end_time = time.perf_counter()

    deserialization_start_time = time.perf_counter()
    deserialized = datatypes.deserialize(serialized)["param_0"]
    deserialization_end_time = time.perf_counter()
    logger.info(f"Deserialized Point3D: {deserialized}")
    logger.info(f"Deserialized Point3D data: {deserialized.data == new_point3d_data}")

    logger.info(
        f"Serialized Point3D to PyArrow in {(serialization_end_time - serialization_start_time) * 1000:.6f} ms."
    )
    logger.info(
        f"Deserialized Point3D from PyArrow in {(deserialization_end_time - deserialization_start_time) * 1000:.6f} ms."
    )


if __name__ == "__main__":
    point3d_example()
