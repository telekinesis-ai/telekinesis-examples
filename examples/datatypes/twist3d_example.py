"""
Example script to demonstrate usage of Twist3D datatype.
"""

import time

import numpy as np
from loguru import logger
import rerun as rr

from telekinesis import datatypes


def twist3d_example():
    """
    Example function to demonstrate usage of Twist3D datatype.
        - Create a Twist3D from linear + angular velocity
        - Access the underlying ndarray through ``.data`` (defensive copy)
        - Inspect ``shape``, ``ndim``, ``dtype`` and ``size``
        - Visualize the Twist3D data using Rerun
        - Replace the wrapped payload via the ``data`` setter
        - Use with numpy to split into linear/angular components
        - Serialize to PyArrow and back
    """

    # Create a Twist3D data: [vx, vy, vz, wx, wy, wz]
    input_twist3d = np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.2], dtype=np.float32)
    logger.info(f"Input twist: {input_twist3d}")
    my_twist3d = datatypes.Twist3D(input_twist3d)
    logger.info(f"Original Twist3D: {my_twist3d}")

    # Access the underlying data
    my_twist3d_data = my_twist3d.data
    my_twist3d_shape = my_twist3d.shape
    my_twist3d_size = my_twist3d.size
    my_twist3d_dtype = my_twist3d.dtype
    my_twist3d_ndim = my_twist3d.ndim
    my_twist3d_numpy = my_twist3d.to_numpy()
    my_twist3d_copy = my_twist3d.copy()

    logger.info(f"Underlying Twist3D data: {my_twist3d_data}")
    logger.info(f"Underlying Twist3D shape: {my_twist3d_shape}")
    logger.info(f"Underlying Twist3D size: {my_twist3d_size}")
    logger.info(f"Underlying Twist3D dtype: {my_twist3d_dtype}")
    logger.info(f"Underlying Twist3D ndim: {my_twist3d_ndim}")
    logger.info(f"Underlying Twist3D numpy array: {my_twist3d_numpy}")
    logger.info(f"Underlying Twist3D object: {my_twist3d_copy}")

    logger.info("Visualizing with Rerun...")
    rr.init("twist3d_example", spawn=True)
    datatypes.visualize(my_twist3d, entity_path="/Twist3D/main", label="Original Twist3D")

    # Update the underlying data
    my_twist3d.data = np.array([0.0, 0.3, 0.0, 0.1, 0.0, 0.0], dtype=np.float32)
    logger.info(f"Updated Twist3D: {my_twist3d}")
    datatypes.visualize(my_twist3d, entity_path="/Twist3D/updated", label="Updated Twist3D")

    # Use with numpy: split the flat array into its linear/angular halves
    linear = my_twist3d_numpy[:3]
    angular = my_twist3d_numpy[3:]
    logger.info(f"Linear velocity (vx, vy, vz): {linear}")
    logger.info(f"Angular velocity (wx, wy, wz): {angular}")

    # Operate on the underlying data with numpy
    linear_speed = np.linalg.norm(linear)
    angular_speed = np.linalg.norm(angular)
    logger.info(f"Linear speed via numpy: {linear_speed}")
    logger.info(f"Angular speed via numpy: {angular_speed}")

    # Serialize to PyArrow and back
    serialization_start_time = time.perf_counter()
    serialized = datatypes.serialize(my_twist3d)
    serialization_end_time = time.perf_counter()

    deserialization_start_time = time.perf_counter()
    deserialized = datatypes.deserialize(serialized)["param_0"]
    deserialization_end_time = time.perf_counter()
    logger.info(f"Deserialized Twist3D: {deserialized}")
    logger.info(
        f"Deserialized Twist3D and original Twist3D are equal: {deserialized == my_twist3d}"
    )

    logger.info(
        f"Serialization time: {(serialization_end_time - serialization_start_time) * 1000} ms"
    )
    logger.info(
        f"Deserialization time: {(deserialization_end_time - deserialization_start_time) * 1000} ms"
    )


if __name__ == "__main__":
    twist3d_example()
