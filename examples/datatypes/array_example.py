"""
Example script to demonstrate usage of Array datatype.
"""

import time

import numpy as np
from loguru import logger
import rerun as rr

from telekinesis import datatypes


def array_example():
    """
    Example function to demonstrate usage of Array datatype.
     - Create an Array data from an ndarray, list, or tuple
     - Access the underlying ndarray through ``.data`` (defensive copy)
     - Inspect ``shape``, ``ndim``, ``dtype`` and ``size``
     - Replace the wrapped payload via the ``data`` setter (any shape/dtype)
     - NumPy interop via ``np.asarray``
    """

    # Create Array data
    input_array = np.arange(12, dtype=np.int32).reshape(3, 4)
    logger.info(f"Input array:\n{input_array}")
    my_array = datatypes.Array(input_array)
    logger.info(f"Original Array: {my_array}")

    # Access the underlying data
    my_array_data = my_array.data
    my_array_shape = my_array.shape
    my_array_size = my_array.size
    my_array_dtype = my_array.dtype
    my_array_ndim = my_array.ndim
    my_array_numpy = my_array.to_numpy()
    my_array_copy = my_array.copy()

    logger.info(f"Underlying Array data: {my_array_data}")
    logger.info(f"Underlying Array shape: {my_array_shape}")
    logger.info(f"Underlying Array size: {my_array_size}")
    logger.info(f"Underlying Array dtype: {my_array_dtype}")
    logger.info(f"Underlying Array ndim: {my_array_ndim}")
    logger.info(f"Underlying Array numpy array: {my_array_numpy}")
    logger.info(f"Underlying Array object: {my_array_copy}")

    logger.info("Visualizing with Rerun...")
    rr.init("array_example", spawn=True)
    datatypes.visualize(my_array, entity_path="/Array")

    # Update the underlying data via the setter
    my_array.data = np.zeros((2, 2), dtype=np.float64)
    logger.info(f"Updated Array: {my_array}")
    datatypes.visualize(my_array, entity_path="/Array/updated")

    # Use with numpy
    my_array_numpy = np.reshape(my_array_numpy, (2, 6))
    logger.info(f"Underlying Array with np.reshape: {my_array_numpy}")

    # Operate on the underlying data with numpy
    my_array_sum = np.sum(my_array_numpy)
    logger.info(f"Sum Array numpy array np.sum: {my_array_sum}")

    # Serialize to PyArrow and back
    serialization_start_time = time.perf_counter()
    my_array_arrow = datatypes.serialize(my_array)
    serialization_end_time = time.perf_counter()

    deserialization_start_time = time.perf_counter()
    my_array_from_arrow = datatypes.deserialize(my_array_arrow)["param_0"]
    deserialization_end_time = time.perf_counter()
    logger.info(f"Deserialized Array from PyArrow: {my_array_from_arrow}")
    logger.info(
        f"Deserialized Array and original Array are equal: {my_array == my_array_from_arrow}"
    )

    logger.info(
        f"Serialization time: {(serialization_end_time - serialization_start_time) * 1000} ms"
    )
    logger.info(
        f"Deserialization time: {(deserialization_end_time - deserialization_start_time) * 1000} ms"
    )


if __name__ == "__main__":
    array_example()
