"""
Example script to demonstrate usage of Transform2D datatype.
"""

import time

import numpy as np
from loguru import logger
import rerun as rr

from telekinesis import datatypes


def transform2d_example():
    """
    Example function to demonstrate usage of Transform2D datatype.
        - Create a Transform2D data (3x3 homogeneous transform)
        - Access the underlying transform data
        - Visualize the Transform2D data using Rerun
        - Update the underlying transform data
        - Operate on the underlying data with numpy
        - Serialize to PyArrow and back
    """
    # Create a Transform2D data (3x3 homogeneous transform: 45 degree rotation + translation)
    theta = np.pi / 4
    matrix = np.array(
        [
            [np.cos(theta), -np.sin(theta), 1.0],
            [np.sin(theta), np.cos(theta), 2.0],
            [0.0, 0.0, 1.0],
        ]
    )
    my_transform2d = datatypes.Transform2D(matrix)
    logger.info(f"Original Transform2D: {my_transform2d}")

    # Access the underlying transform data
    my_transform2d_data = my_transform2d.data
    my_transform2d_shape = my_transform2d.shape
    my_transform2d_size = my_transform2d.size
    my_transform2d_dtype = my_transform2d.dtype
    my_transform2d_ndim = my_transform2d.ndim
    my_transform2d_numpy = my_transform2d.to_numpy()
    my_transform2d_copy = my_transform2d.copy()

    logger.info(f"Underlying Transform2D data: {my_transform2d_data}")
    logger.info(f"Underlying Transform2D data shape: {my_transform2d_shape}")
    logger.info(f"Underlying Transform2D data size: {my_transform2d_size}")
    logger.info(f"Underlying Transform2D data dtype: {my_transform2d_dtype}")
    logger.info(f"Underlying Transform2D data ndim: {my_transform2d_ndim}")
    logger.info(f"Underlying Transform2D data as numpy array: {my_transform2d_numpy}")
    logger.info(f"Underlying Transform2D object: {my_transform2d_copy}")

    logger.info("Visualizing with Rerun...")
    rr.init("transform2d_example", spawn=True)
    datatypes.visualize(
        my_transform2d,
        entity_path="/Transform2D/main",
        label="My Transform2D",
    )

    # Update the my_transform2d_data
    theta_updated = np.pi / 2
    new_matrix = np.array(
        [
            [np.cos(theta_updated), -np.sin(theta_updated), 1.5],
            [np.sin(theta_updated), np.cos(theta_updated), 2.5],
            [0.0, 0.0, 1.0],
        ]
    )
    my_transform2d.data = new_matrix
    logger.info(f"Updated Transform2D: {my_transform2d}")
    datatypes.visualize(
        my_transform2d,
        entity_path="/Transform2D/updated",
        label="Updated Transform2D",
    )

    # Operate on the underlying data with numpy - Add to the last column of the transform matrix
    my_transform2d_sum = np.array([1, 1, 0]) + my_transform2d_numpy
    logger.info(f"Sum of Transform2D with numpy array : {my_transform2d_sum}")

    # Serialize to PyArrow and back
    serialization_start_time = time.perf_counter()
    serialized = datatypes.serialize(my_transform2d)
    serialization_end_time = time.perf_counter()

    deserialization_start_time = time.perf_counter()
    deserialized_transform2d = datatypes.deserialize(serialized)["param_0"]
    logger.info(f"Deserialized Transform2D: {deserialized_transform2d}")
    logger.info(
        f"Deserialized Transform2D is equal to original: {deserialized_transform2d == my_transform2d}"
    )
    deserialization_end_time = time.perf_counter()

    logger.info(
        f"Serialization time: {(serialization_end_time - serialization_start_time) * 1000} ms"
    )
    logger.info(
        f"Deserialization time: {(deserialization_end_time - deserialization_start_time) * 1000} ms"
    )


if __name__ == "__main__":
    transform2d_example()
