"""
Example script to demonstrate usage of Position3D datatype.
"""

import time

import numpy as np
from loguru import logger
import rerun as rr

from telekinesis import datatypes


def position3d_example():
    """
    Example function to demonstrate usage of Position3D datatype.
        - Create a Position3D data
        - Access the underlying position data
        - Visualize the Position3D data using Rerun
        - Update the underlying position data
        - Operate on the underlying data with numpy
        - Serialize to PyArrow and back
    """
    # Create a Position3D data
    position = [1.0, 2.0, 3.0]
    my_position3d = datatypes.Position3D(position)
    logger.info(f"Original Position3D: {my_position3d}")

    # Access the underlying position data
    my_position3d_data = my_position3d.data
    my_position3d_shape = my_position3d.shape
    my_position3d_size = my_position3d.size
    my_position3d_dtype = my_position3d.dtype
    my_position3d_ndim = my_position3d.ndim
    my_position3d_numpy = my_position3d.to_numpy()
    my_position3d_copy = my_position3d.copy()

    logger.info(f"Underlying Position3D data: {my_position3d_data}")
    logger.info(f"Underlying Position3D shape: {my_position3d_shape}")
    logger.info(f"Underlying Position3D size: {my_position3d_size}")
    logger.info(f"Underlying Position3D dtype: {my_position3d_dtype}")
    logger.info(f"Underlying Position3D ndim: {my_position3d_ndim}")
    logger.info(f"Underlying Position3D numpy array: {my_position3d_numpy}")
    logger.info(f"Underlying Position3D object: {my_position3d_copy}")

    # Visualize the Position3D data using Rerun
    logger.info("Visualizing with Rerun...")
    rr.init("position3d_example", spawn=True)
    datatypes.visualize(my_position3d, entity_path="/Position3D", label="My Position3D")

    # Update the my_position3d_data
    new_position3d_data = [4.0, 5.0, 6.0]
    my_position3d.data = new_position3d_data
    logger.info(f"Updated Position3D: {my_position3d}")
    datatypes.visualize(
        my_position3d, entity_path="/Position3D/updated", label="Updated Position3D"
    )

    # Operate on the underlying data with numpy - Subtract from the position
    my_position3d_diff = my_position3d - np.array([1.0, 1.0, 1.0])
    logger.info(f"Difference of Position3D with numpy array: {my_position3d_diff}")

    # Serialize to PyArrow and back
    serialization_start_time = time.perf_counter()
    serialized = datatypes.serialize(my_position3d)
    serialization_end_time = time.perf_counter()

    deserialization_start_time = time.perf_counter()
    deserialized = datatypes.deserialize(serialized)["param_0"]
    deserialization_end_time = time.perf_counter()
    logger.info(f"Deserialized Position3D: {deserialized}")
    logger.info(f"Deserialized Position3D data: {deserialized.data == new_position3d_data}")

    logger.info(
        f"Serialized Position3D to PyArrow in {(serialization_end_time - serialization_start_time) * 1000:.6f} ms."
    )
    logger.info(
        f"Deserialized Position3D from PyArrow in {(deserialization_end_time - deserialization_start_time) * 1000:.6f} ms."
    )


if __name__ == "__main__":
    position3d_example()
