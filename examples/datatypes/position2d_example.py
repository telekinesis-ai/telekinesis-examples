"""
Example script to demonstrate usage of Position2D datatype.
"""

import time

import numpy as np
from loguru import logger
import rerun as rr

from telekinesis import datatypes


def position2d_example():
    """
    Example function to demonstrate usage of Position2D datatype.
        - Create a Position2D data
        - Access the underlying position data
        - Visualize the Position2D data using Rerun
        - Update the underlying position data
        - Operate on the underlying data with numpy
        - Serialize to PyArrow and back
    """
    # Create a Position2D data can be list or numpy array of shape (2,)
    position = [10.0, 20.0]
    my_position2d = datatypes.Position2D(position)
    logger.info(f"Original Position2D: {my_position2d}")

    # Access the underlying position data
    my_position2d_data = my_position2d.data
    my_position2d_shape = my_position2d.shape
    my_position2d_size = my_position2d.size
    my_position2d_dtype = my_position2d.dtype
    my_position2d_ndim = my_position2d.ndim
    my_position2d_numpy = my_position2d.to_numpy()
    my_position2d_copy = my_position2d.copy()

    logger.info(f"Underlying Position2D data: {my_position2d_data}")
    logger.info(f"Underlying Position2D shape: {my_position2d_shape}")
    logger.info(f"Underlying Position2D size: {my_position2d_size}")
    logger.info(f"Underlying Position2D dtype: {my_position2d_dtype}")
    logger.info(f"Underlying Position2D ndim: {my_position2d_ndim}")
    logger.info(f"Underlying Position2D numpy array: {my_position2d_numpy}")
    logger.info(f"Underlying Position2D object: {my_position2d_copy}")

    # Visualize the Position2D data using Rerun
    logger.info("Visualizing with Rerun...")
    rr.init("position2d_example", spawn=True)
    datatypes.visualize(my_position2d, entity_path="/Position2D", label="My Position2D")

    # Update the my_position2d_data
    new_position2d_data = [30.0, 40.0]
    my_position2d.data = new_position2d_data
    logger.info(f"Updated Position2D: {my_position2d}")
    datatypes.visualize(
        my_position2d, entity_path="/Position2D/updated", label="Updated Position2D"
    )

    # Operate on the underlying data with numpy - Add to the position
    my_position2d_sum = my_position2d + np.array([5.0, 10.0])
    logger.info(f"Sum of Position2D with numpy array: {my_position2d_sum}")

    # Serialize to PyArrow and back
    serialization_start_time = time.perf_counter()
    serialized = datatypes.serialize(my_position2d)
    serialization_end_time = time.perf_counter()

    deserialization_start_time = time.perf_counter()
    deserialized = datatypes.deserialize(serialized)["param_0"]
    deserialization_end_time = time.perf_counter()
    logger.info(f"Deserialized Position2D: {deserialized}")
    logger.info(f"Deserialized Position2D data: {deserialized.data == new_position2d_data}")

    logger.info(
        f"Serialized Position2D to PyArrow in {(serialization_end_time - serialization_start_time) * 1000:.6f} ms."
    )
    logger.info(
        f"Deserialized Position2D from PyArrow in {(deserialization_end_time - deserialization_start_time) * 1000:.6f} ms."
    )


if __name__ == "__main__":
    position2d_example()
