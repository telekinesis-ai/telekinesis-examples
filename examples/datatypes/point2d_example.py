"""
Example script to demonstrate usage of Point2D datatype.
"""

import time

import numpy as np
from loguru import logger
import rerun as rr

from telekinesis import datatypes


def point2d_example():
    """
    Example function to demonstrate usage of Point2D datatype.
        - Create a Point2D data
        - Access the underlying point data
        - Visualize the Point2D data using Rerun
        - Update the underlying point data
        - Operate on the underlying data with numpy
        - Serialize to PyArrow and back
    """
    # Create a Point2D data can be list or numpy array of shape (2,)
    point = [1.0, 2.0]
    my_point2d = datatypes.Point2D(point)
    logger.info(f"Original Point2D: {my_point2d}")

    # Access the underlying point data
    my_point2d_data = my_point2d.data
    my_point2d_shape = my_point2d.shape
    my_point2d_size = my_point2d.size
    my_point2d_dtype = my_point2d.dtype
    my_point2d_ndim = my_point2d.ndim
    my_point2d_numpy = my_point2d.to_numpy()
    my_point2d_copy = my_point2d.copy()

    logger.info(f"Underlying Point2D data: {my_point2d_data}")
    logger.info(f"Underlying Point2D shape: {my_point2d_shape}")
    logger.info(f"Underlying Point2D size: {my_point2d_size}")
    logger.info(f"Underlying Point2D dtype: {my_point2d_dtype}")
    logger.info(f"Underlying Point2D ndim: {my_point2d_ndim}")
    logger.info(f"Underlying Point2D numpy array: {my_point2d_numpy}")
    logger.info(f"Underlying Point2D object: {my_point2d_copy}")

    # Visualize the Point2D data using Rerun
    logger.info("Visualizing with Rerun...")
    rr.init("point2d_example", spawn=True)
    datatypes.visualize(my_point2d, entity_path="/Point2D", label="My Point2D")

    # Update the my_point2d_data
    new_point2d_data = [3.0, 4.0]
    my_point2d.data = new_point2d_data
    logger.info(f"Updated Point2D: {my_point2d}")
    datatypes.visualize(my_point2d, entity_path="/Point2D/updated", label="Updated Point2D")

    # Operate on the underlying data with numpy - add, subtract, multiply, divide
    my_point2d_sum = my_point2d + np.array([1.0, 1.0])
    my_point2d_diff = my_point2d - np.array([1.0, 1.0])
    my_point2d_prod = my_point2d * np.array(2.0)
    my_point2d_quot = my_point2d / np.array(2.0)
    logger.info(f"Sum of Point2D with numpy array: {my_point2d_sum}")
    logger.info(f"Difference of Point2D with numpy array: {my_point2d_diff}")
    logger.info(f"Product of Point2D with scalar: {my_point2d_prod}")
    logger.info(f"Quotient of Point2D with scalar: {my_point2d_quot}")

    # Serialize to PyArrow and back
    serialization_start_time = time.perf_counter()
    serialized = datatypes.serialize(my_point2d)
    serialization_end_time = time.perf_counter()

    deserialization_start_time = time.perf_counter()
    deserialized = datatypes.deserialize(serialized)["param_0"]
    deserialization_end_time = time.perf_counter()
    logger.info(f"Deserialized Point2D: {deserialized}")
    logger.info(f"Deserialized Point2D data: {deserialized.data == new_point2d_data}")

    logger.info(
        f"Serialized Point2D to PyArrow in {(serialization_end_time - serialization_start_time) * 1000:.6f} ms."
    )
    logger.info(
        f"Deserialized Point2D from PyArrow in {(deserialization_end_time - deserialization_start_time) * 1000:.6f} ms."
    )


if __name__ == "__main__":
    point2d_example()
