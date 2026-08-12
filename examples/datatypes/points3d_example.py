"""
Example script to demonstrate usage of Points3D datatype.
"""

import time

import numpy as np
from loguru import logger
import rerun as rr

from telekinesis import datatypes


def points3d_example():
    """
    Example function to demonstrate usage of Points3D datatype.
        - Create a Points3D data
        - Access the underlying points data
        - Visualize the Points3D data using Rerun
        - Update the underlying points data
        - Operate on the underlying data with numpy
        - Serialize to PyArrow and back
        - Construct an empty (zero-row) batch
    """
    # Create a Points3D data can be list or numpy array of shape (N, 3)
    points = [[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]]
    my_points3d = datatypes.Points3D(points)
    logger.info(f"Original Points3D: {my_points3d}")

    # Access the underlying points data
    my_points3d_data = my_points3d.data
    my_points3d_shape = my_points3d.shape
    my_points3d_size = my_points3d.size
    my_points3d_dtype = my_points3d.dtype
    my_points3d_ndim = my_points3d.ndim
    my_points3d_numpy = my_points3d.to_numpy()
    my_points3d_copy = my_points3d.copy()

    logger.info(f"Underlying Points3D data: {my_points3d_data}")
    logger.info(f"Underlying Points3D shape: {my_points3d_shape}")
    logger.info(f"Underlying Points3D size: {my_points3d_size}")
    logger.info(f"Underlying Points3D dtype: {my_points3d_dtype}")
    logger.info(f"Underlying Points3D ndim: {my_points3d_ndim}")
    logger.info(f"Underlying Points3D numpy array: {my_points3d_numpy}")
    logger.info(f"Underlying Points3D object: {my_points3d_copy}")

    # Visualize the Points3D data using Rerun
    logger.info("Visualizing with Rerun...")
    rr.init("points3d_example", spawn=True)
    datatypes.visualize(my_points3d, entity_path="/Points3D", label=["Point 1", "Point 2"])

    # Update the my_points3d_data
    new_points3d_data = [[70.0, 80.0, 90.0], [100.0, 110.0, 120.0]]
    my_points3d.data = new_points3d_data
    logger.info(f"Updated Points3D: {my_points3d}")
    datatypes.visualize(
        my_points3d,
        entity_path="/Points3D/updated",
        label=["Updated Point 1", "Updated Point 2"],
    )

    # Operate on the underlying data with numpy - add, subtract, multiply, divide
    my_points3d_sum = my_points3d + np.array([1.0, 1.0, 1.0])
    my_points3d_diff = my_points3d - np.array([1.0, 1.0, 1.0])
    my_points3d_prod = my_points3d * np.array(2.0)
    my_points3d_quot = my_points3d / np.array(2.0)
    logger.info(f"Sum of Points3D with numpy array: {my_points3d_sum}")
    logger.info(f"Difference of Points3D with numpy array: {my_points3d_diff}")
    logger.info(f"Product of Points3D with scalar: {my_points3d_prod}")
    logger.info(f"Quotient of Points3D with scalar: {my_points3d_quot}")

    # Serialize to PyArrow and back
    serialization_start_time = time.perf_counter()
    serialized = datatypes.serialize(my_points3d)
    serialization_end_time = time.perf_counter()

    deserialization_start_time = time.perf_counter()
    deserialized = datatypes.deserialize(serialized)["param_0"]
    deserialization_end_time = time.perf_counter()
    logger.info(f"Deserialized Points3D: {deserialized}")
    logger.info(f"Deserialized Points3D data: {deserialized.data == new_points3d_data}")

    logger.info(
        f"Serialized Points3D to PyArrow in {(serialization_end_time - serialization_start_time) * 1000:.6f} ms."
    )
    logger.info(
        f"Deserialized Points3D from PyArrow in {(deserialization_end_time - deserialization_start_time) * 1000:.6f} ms."
    )

    # Points3D also accepts an empty (zero-row) batch, since shape_spec's leading
    # axis (None) allows N=0. A plain `[]` won't work since it has no second axis,
    # so an explicitly-shaped empty array is required.
    empty_points3d = datatypes.Points3D(np.empty((0, 3), dtype=np.float32))
    logger.info(f"Empty Points3D: {empty_points3d}")
    logger.info(f"Empty Points3D shape: {empty_points3d.shape}")


if __name__ == "__main__":
    points3d_example()
