"""
Example script to demonstrate usage of Wrench3D datatype.
"""

import time

import numpy as np
from loguru import logger
import rerun as rr

from telekinesis import datatypes


def wrench3d_example():
    """
    Example function to demonstrate usage of Wrench3D datatype.
        - Create a Wrench3D from force + torque
        - Access the underlying ndarray through ``.data`` (defensive copy)
        - Inspect ``shape``, ``ndim``, ``dtype`` and ``size``
        - Visualize the Wrench3D data using Rerun
        - Replace the wrapped payload via the ``data`` setter
        - Use with numpy to split into force/torque components
        - Serialize to PyArrow and back
    """

    # Create a Wrench3D data: [fx, fy, fz, tx, ty, tz]
    input_wrench3d = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.5], dtype=np.float32)
    logger.info(f"Input wrench: {input_wrench3d}")
    my_wrench3d = datatypes.Wrench3D(input_wrench3d)
    logger.info(f"Original Wrench3D: {my_wrench3d}")

    # Access the underlying data
    my_wrench3d_data = my_wrench3d.data
    my_wrench3d_shape = my_wrench3d.shape
    my_wrench3d_size = my_wrench3d.size
    my_wrench3d_dtype = my_wrench3d.dtype
    my_wrench3d_ndim = my_wrench3d.ndim
    my_wrench3d_numpy = my_wrench3d.to_numpy()
    my_wrench3d_copy = my_wrench3d.copy()

    logger.info(f"Underlying Wrench3D data: {my_wrench3d_data}")
    logger.info(f"Underlying Wrench3D shape: {my_wrench3d_shape}")
    logger.info(f"Underlying Wrench3D size: {my_wrench3d_size}")
    logger.info(f"Underlying Wrench3D dtype: {my_wrench3d_dtype}")
    logger.info(f"Underlying Wrench3D ndim: {my_wrench3d_ndim}")
    logger.info(f"Underlying Wrench3D numpy array: {my_wrench3d_numpy}")
    logger.info(f"Underlying Wrench3D object: {my_wrench3d_copy}")

    logger.info("Visualizing with Rerun...")
    rr.init("wrench3d_example", spawn=True)
    datatypes.visualize(my_wrench3d, entity_path="/Wrench3D", label="Original Wrench3D")

    # Update the underlying data
    my_wrench3d.data = np.array([0.0, 2.0, 0.0, 0.0, 0.0, 1.5], dtype=np.float32)
    logger.info(f"Updated Wrench3D: {my_wrench3d}")
    datatypes.visualize(my_wrench3d, entity_path="/Wrench3D/updated", label="Updated Wrench3D")

    # Use with numpy: split the flat array into its force/torque halves
    force = my_wrench3d_numpy[:3]
    torque = my_wrench3d_numpy[3:]
    logger.info(f"Force (fx, fy, fz): {force}")
    logger.info(f"Torque (tx, ty, tz): {torque}")

    # Operate on the underlying data with numpy
    force_magnitude = np.linalg.norm(force)
    torque_magnitude = np.linalg.norm(torque)
    logger.info(f"Force magnitude via numpy: {force_magnitude}")
    logger.info(f"Torque magnitude via numpy: {torque_magnitude}")

    # Serialize to PyArrow and back
    serialization_start_time = time.perf_counter()
    serialized = datatypes.serialize(my_wrench3d)
    serialization_end_time = time.perf_counter()

    deserialization_start_time = time.perf_counter()
    deserialized = datatypes.deserialize(serialized)["param_0"]
    deserialization_end_time = time.perf_counter()
    logger.info(f"Deserialized Wrench3D: {deserialized}")
    logger.info(
        f"Deserialized Wrench3D and original Wrench3D are equal: {deserialized == my_wrench3d}"
    )

    logger.info(
        f"Serialization time: {(serialization_end_time - serialization_start_time) * 1000} ms"
    )
    logger.info(
        f"Deserialization time: {(deserialization_end_time - deserialization_start_time) * 1000} ms"
    )


if __name__ == "__main__":
    wrench3d_example()
