"""
Example script to demonstrate usage of Covariance6x6 datatype.
"""

import time

import numpy as np
from loguru import logger
import rerun as rr

from telekinesis import datatypes


def covariance6x6_example():
    """
    Example function to demonstrate usage of Covariance6x6 datatype.
        - Create a Covariance6x6 data from a symmetric positive semi-definite matrix
        - Access the underlying ndarray through ``.data`` (defensive copy)
        - Inspect ``shape``, ``ndim``, ``dtype`` and ``size``
        - Replace the wrapped payload via the ``data`` setter (must stay symmetric PSD)
        - NumPy interop: check symmetry, compute eigenvalues, and read off per-axis variances
        - Serialize to PyArrow and back
    """

    # Create Covariance6x6 data
    input_covariance = np.diag([1.0, 2.0, 3.0, 0.5, 0.5, 0.5]).astype(np.float32)
    logger.info(f"Input covariance:\n{input_covariance}")
    my_covariance6x6 = datatypes.Covariance6x6(input_covariance)
    logger.info(f"Original Covariance6x6: {my_covariance6x6}")

    # Access the underlying data
    my_covariance6x6_data = my_covariance6x6.data
    my_covariance6x6_shape = my_covariance6x6.shape
    my_covariance6x6_size = my_covariance6x6.size
    my_covariance6x6_dtype = my_covariance6x6.dtype
    my_covariance6x6_ndim = my_covariance6x6.ndim
    my_covariance6x6_numpy = my_covariance6x6.to_numpy()
    my_covariance6x6_copy = my_covariance6x6.copy()

    logger.info(f"Underlying Covariance6x6 data:\n{my_covariance6x6_data}")
    logger.info(f"Underlying Covariance6x6 shape: {my_covariance6x6_shape}")
    logger.info(f"Underlying Covariance6x6 size: {my_covariance6x6_size}")
    logger.info(f"Underlying Covariance6x6 dtype: {my_covariance6x6_dtype}")
    logger.info(f"Underlying Covariance6x6 ndim: {my_covariance6x6_ndim}")
    logger.info(f"Underlying Covariance6x6 numpy array:\n{my_covariance6x6_numpy}")
    logger.info(f"Underlying Covariance6x6 object: {my_covariance6x6_copy}")

    logger.info("Visualizing with Rerun...")
    rr.init("covariance6x6_example", spawn=True)
    datatypes.visualize(my_covariance6x6, entity_path="/Covariance6x6")

    # Update the underlying data via the setter (must remain symmetric PSD)
    my_covariance6x6.data = np.eye(6, dtype=np.float32) * 2.0
    logger.info(f"Updated Covariance6x6:\n{my_covariance6x6.data}")
    datatypes.visualize(my_covariance6x6, entity_path="/Covariance6x6/updated")

    # Use with numpy - check symmetry, eigenvalues, and per-axis variances
    is_symmetric = np.allclose(my_covariance6x6.data, my_covariance6x6.data.T)
    eigenvalues = np.linalg.eigvalsh(my_covariance6x6.data)
    variances = np.diag(my_covariance6x6.data)
    logger.info(f"Is symmetric (np.allclose with transpose): {is_symmetric}")
    logger.info(f"Eigenvalues (np.linalg.eigvalsh): {eigenvalues}")
    logger.info(f"Per-axis variances (np.diag): {variances}")

    # Serialize to PyArrow and back
    serialization_start_time = time.perf_counter()
    my_covariance6x6_arrow = datatypes.serialize(my_covariance6x6)
    serialization_end_time = time.perf_counter()

    deserialization_start_time = time.perf_counter()
    my_covariance6x6_from_arrow = datatypes.deserialize(my_covariance6x6_arrow)["param_0"]
    deserialization_end_time = time.perf_counter()
    logger.info(f"Deserialized Covariance6x6 from PyArrow:\n{my_covariance6x6_from_arrow.data}")
    logger.info(
        f"Deserialized Covariance6x6 and original Covariance6x6 are equal: "
        f"{my_covariance6x6 == my_covariance6x6_from_arrow}"
    )

    logger.info(
        f"Serialization time: {(serialization_end_time - serialization_start_time) * 1000} ms"
    )
    logger.info(
        f"Deserialization time: {(deserialization_end_time - deserialization_start_time) * 1000} ms"
    )


if __name__ == "__main__":
    covariance6x6_example()
