"""
Example script to demonstrate usage of Pose2D datatype.
"""

import time

import numpy as np
from loguru import logger
import rerun as rr

from telekinesis import datatypes


def pose2d_example():
    """
    Example function to demonstrate usage of Pose2D datatype.
        - Create a Pose2D data
        - Access the underlying pose data
        - Visualize the Pose2D data using Rerun
        - Update the underlying pose data
        - Operate on the underlying data with numpy
        - Serialize to PyArrow and back
    """
    # Create a Pose2D data can be list or numpy array of shape (3,): [x, y, theta]
    pose = [1.0, 2.0, 0.5]
    my_pose2d = datatypes.Pose2D(pose)
    logger.info(f"Original Pose2D: {my_pose2d}")

    # Access the underlying pose data
    my_pose2d_data = my_pose2d.data
    my_pose2d_shape = my_pose2d.shape
    my_pose2d_size = my_pose2d.size
    my_pose2d_dtype = my_pose2d.dtype
    my_pose2d_ndim = my_pose2d.ndim
    my_pose2d_numpy = my_pose2d.to_numpy()
    my_pose2d_copy = my_pose2d.copy()

    logger.info(f"Underlying Pose2D data: {my_pose2d_data}")
    logger.info(f"Underlying Pose2D shape: {my_pose2d_shape}")
    logger.info(f"Underlying Pose2D size: {my_pose2d_size}")
    logger.info(f"Underlying Pose2D dtype: {my_pose2d_dtype}")
    logger.info(f"Underlying Pose2D ndim: {my_pose2d_ndim}")
    logger.info(f"Underlying Pose2D numpy array: {my_pose2d_numpy}")
    logger.info(f"Underlying Pose2D object: {my_pose2d_copy}")

    # Visualize the Pose2D data using Rerun
    logger.info("Visualizing with Rerun...")
    rr.init("pose2d_example", spawn=True)
    datatypes.visualize(my_pose2d, entity_path="/Pose2D", label="My Pose2D")

    # Update the my_pose2d_data
    new_pose2d_data = [3.0, 4.0, 1.0]
    my_pose2d.data = new_pose2d_data
    logger.info(f"Updated Pose2D: {my_pose2d}")
    datatypes.visualize(my_pose2d, entity_path="/Pose2D/updated", label="Updated Pose2D")

    # Operate on the underlying data with numpy - translate x, y (theta unchanged)
    my_pose2d_sum = my_pose2d + np.array([1.0, 1.0, 0.0])
    logger.info(f"Sum of Pose2D with numpy array: {my_pose2d_sum}")

    # Serialize to PyArrow and back
    serialization_start_time = time.perf_counter()
    serialized = datatypes.serialize(my_pose2d)
    serialization_end_time = time.perf_counter()

    deserialization_start_time = time.perf_counter()
    deserialized = datatypes.deserialize(serialized)["param_0"]
    deserialization_end_time = time.perf_counter()
    logger.info(f"Deserialized Pose2D: {deserialized}")
    logger.info(f"Deserialized Pose2D data: {deserialized.data == new_pose2d_data}")

    logger.info(
        f"Serialized Pose2D to PyArrow in {(serialization_end_time - serialization_start_time) * 1000:.6f} ms."
    )
    logger.info(
        f"Deserialized Pose2D from PyArrow in {(deserialization_end_time - deserialization_start_time) * 1000:.6f} ms."
    )


if __name__ == "__main__":
    pose2d_example()
