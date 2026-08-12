"""
Example script to demonstrate usage of Quaternion datatype.
"""

import time

import numpy as np
from loguru import logger
import rerun as rr
from scipy.spatial.transform import Rotation

from telekinesis import datatypes


def quaternion_example():
    """
    Example function to demonstrate usage of Quaternion datatype.
     - Create a Quaternion data (unit quaternion, scalar-last [qx, qy, qz, qw])
     - Access the underlying ndarray through ``.data`` (defensive copy)
     - Inspect ``shape``, ``ndim``, ``dtype`` and ``size``
     - Replace the wrapped payload via the ``data`` setter (must stay unit-norm)
     - NumPy/SciPy interop: verify unit norm and convert to a rotation matrix
     - Serialize to PyArrow and back
    """

    # Create a Quaternion data (identity-adjacent rotation: [qx, qy, qz, qw])
    my_quaternion = datatypes.Quaternion([0.4619398, 0.1913417, 0.4619398, 0.7325378])
    logger.info(f"Original Quaternion: {my_quaternion}")

    # Access the underlying data
    my_quaternion_data = my_quaternion.data
    my_quaternion_shape = my_quaternion.shape
    my_quaternion_size = my_quaternion.size
    my_quaternion_dtype = my_quaternion.dtype
    my_quaternion_ndim = my_quaternion.ndim
    my_quaternion_numpy = my_quaternion.to_numpy()
    my_quaternion_copy = my_quaternion.copy()

    logger.info(f"Underlying Quaternion data: {my_quaternion_data}")
    logger.info(f"Underlying Quaternion shape: {my_quaternion_shape}")
    logger.info(f"Underlying Quaternion size: {my_quaternion_size}")
    logger.info(f"Underlying Quaternion dtype: {my_quaternion_dtype}")
    logger.info(f"Underlying Quaternion ndim: {my_quaternion_ndim}")
    logger.info(f"Underlying Quaternion numpy array: {my_quaternion_numpy}")
    logger.info(f"Underlying Quaternion object: {my_quaternion_copy}")

    logger.info("Visualizing with Rerun...")
    rr.init("quaternion_example", spawn=True)
    datatypes.visualize(my_quaternion, entity_path="/Quaternion", label="My Quaternion")

    # Update the underlying data via the setter (must remain unit-norm)
    my_quaternion.data = [0.0, 0.0, 0.7071068, 0.7071068]  # 90 deg rotation about Z
    logger.info(f"Updated Quaternion: {my_quaternion}")
    datatypes.visualize(
        my_quaternion, entity_path="/Quaternion/updated", label="Updated Quaternion"
    )

    # Use with numpy/scipy - verify unit norm and convert to a rotation matrix
    quaternion_norm = np.linalg.norm(my_quaternion.data)
    rotation_matrix = Rotation.from_quat(my_quaternion.data).as_matrix()
    logger.info(f"Quaternion norm (np.linalg.norm): {quaternion_norm}")
    logger.info(f"Equivalent rotation matrix (scipy Rotation):\n{rotation_matrix}")

    # Serialize to PyArrow and back
    serialization_start_time = time.perf_counter()
    my_quaternion_arrow = datatypes.serialize(my_quaternion)
    serialization_end_time = time.perf_counter()

    deserialization_start_time = time.perf_counter()
    my_quaternion_from_arrow = datatypes.deserialize(my_quaternion_arrow)["param_0"]
    deserialization_end_time = time.perf_counter()
    logger.info(f"Deserialized Quaternion from PyArrow: {my_quaternion_from_arrow}")
    logger.info(
        f"Deserialized Quaternion and original Quaternion are equal: "
        f"{my_quaternion == my_quaternion_from_arrow}"
    )

    logger.info(
        f"Serialization time: {(serialization_end_time - serialization_start_time) * 1000} ms"
    )
    logger.info(
        f"Deserialization time: {(deserialization_end_time - deserialization_start_time) * 1000} ms"
    )


if __name__ == "__main__":
    quaternion_example()
