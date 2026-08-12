"""
Example script to demonstrate usage of Points2D datatype.
"""

import time

import numpy as np
from loguru import logger
import rerun as rr

from telekinesis import datatypes


def points2d_example():
    """
    Example function to demonstrate usage of Points2D datatype.
        - Create a Points2D data
        - Access the underlying points data
        - Visualize the Points2D data using Rerun
        - Update the underlying points data
        - Operate on the underlying data with numpy
        - Serialize to PyArrow and back
        - Construct an empty (zero-row) batch
    """
    # Create a Points2D data can be list or numpy array of shape (N, 2)
    points = [[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]]
    my_points2d = datatypes.Points2D(points)
    logger.info(f"Original Points2D: {my_points2d}")

    # Access the underlying points data
    my_points2d_data = my_points2d.data
    my_points2d_shape = my_points2d.shape
    my_points2d_size = my_points2d.size
    my_points2d_dtype = my_points2d.dtype
    my_points2d_ndim = my_points2d.ndim
    my_points2d_numpy = my_points2d.to_numpy()
    my_points2d_copy = my_points2d.copy()

    logger.info(f"Underlying Points2D data: {my_points2d_data}")
    logger.info(f"Underlying Points2D shape: {my_points2d_shape}")
    logger.info(f"Underlying Points2D size: {my_points2d_size}")
    logger.info(f"Underlying Points2D dtype: {my_points2d_dtype}")
    logger.info(f"Underlying Points2D ndim: {my_points2d_ndim}")
    logger.info(f"Underlying Points2D numpy array: {my_points2d_numpy}")
    logger.info(f"Underlying Points2D object: {my_points2d_copy}")

    # Visualize the Points2D data using Rerun
    logger.info("Visualizing with Rerun...")
    rr.init("points2d_example", spawn=True)
    datatypes.visualize(
        my_points2d, entity_path="/Points2D", label=["Point 1", "Point 2", "Point 3"]
    )

    # Update the my_points2d_data
    new_points2d_data = [[70.0, 80.0], [90.0, 100.0], [110.0, 120.0]]
    my_points2d.data = new_points2d_data
    logger.info(f"Updated Points2D: {my_points2d}")
    datatypes.visualize(
        my_points2d,
        entity_path="/Points2D/updated",
        label=["Updated Point 1", "Updated Point 2", "Updated Point 3"],
    )

    # Operate on the underlying data with numpy - add, subtract, multiply, divide
    my_points2d_sum = my_points2d + np.array([1.0, 1.0])
    my_points2d_diff = my_points2d - np.array([1.0, 1.0])
    my_points2d_prod = my_points2d * np.array(2.0)
    my_points2d_quot = my_points2d / np.array(2.0)
    logger.info(f"Sum of Points2D with numpy array: {my_points2d_sum}")
    logger.info(f"Difference of Points2D with numpy array: {my_points2d_diff}")
    logger.info(f"Product of Points2D with scalar: {my_points2d_prod}")
    logger.info(f"Quotient of Points2D with scalar: {my_points2d_quot}")

    # Serialize to PyArrow and back
    serialization_start_time = time.perf_counter()
    serialized = datatypes.serialize(my_points2d)
    serialization_end_time = time.perf_counter()

    deserialization_start_time = time.perf_counter()
    deserialized = datatypes.deserialize(serialized)["param_0"]
    deserialization_end_time = time.perf_counter()
    logger.info(f"Deserialized Points2D: {deserialized}")
    logger.info(f"Deserialized Points2D data: {deserialized.data == new_points2d_data}")

    logger.info(
        f"Serialized Points2D to PyArrow in {(serialization_end_time - serialization_start_time) * 1000:.6f} ms."
    )
    logger.info(
        f"Deserialized Points2D from PyArrow in {(deserialization_end_time - deserialization_start_time) * 1000:.6f} ms."
    )

    # Points2D also accepts an empty (zero-row) batch, since shape_spec's leading
    # axis (None) allows N=0. A plain `[]` won't work since it has no second axis,
    # so an explicitly-shaped empty array is required.
    empty_points2d = datatypes.Points2D(np.empty((0, 2), dtype=np.float32))
    logger.info(f"Empty Points2D: {empty_points2d}")
    logger.info(f"Empty Points2D shape: {empty_points2d.shape}")


if __name__ == "__main__":
    points2d_example()
