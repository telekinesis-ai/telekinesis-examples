"""
Example script to demonstrate usage of EigenValues datatype.
"""

import time

import numpy as np
from loguru import logger
import rerun as rr

from telekinesis import datatypes


def eigenvalues_example():
    """
    Example function to demonstrate usage of EigenValues datatype.
     - Create EigenValues from the output of np.linalg.eigh
     - Access the underlying eigenvalues through .data (defensive copy)
     - Inspect shape, ndim, dtype and size
     - Visualize the EigenValues data using Rerun
     - Replace the wrapped payload via the data setter
     - Operate on the underlying data with numpy
     - Serialize to PyArrow and back
    """
    # Create EigenValues from a symmetric matrix decomposition
    input_matrix = np.array([[2.0, 1.0], [1.0, 2.0]], dtype=np.float32)
    w, _ = np.linalg.eigh(input_matrix)
    logger.info(f"Input eigenvalues: {w}")
    my_eigenvalues = datatypes.EigenValues(w)
    logger.info(f"Original EigenValues: {my_eigenvalues}")

    # Access the underlying data
    my_eigenvalues_data = my_eigenvalues.data
    my_eigenvalues_shape = my_eigenvalues.shape
    my_eigenvalues_size = my_eigenvalues.size
    my_eigenvalues_dtype = my_eigenvalues.dtype
    my_eigenvalues_ndim = my_eigenvalues.ndim
    my_eigenvalues_numpy = my_eigenvalues.to_numpy()
    my_eigenvalues_copy = my_eigenvalues.copy()

    logger.info(f"Underlying EigenValues data: {my_eigenvalues_data}")
    logger.info(f"Underlying EigenValues shape: {my_eigenvalues_shape}")
    logger.info(f"Underlying EigenValues size: {my_eigenvalues_size}")
    logger.info(f"Underlying EigenValues dtype: {my_eigenvalues_dtype}")
    logger.info(f"Underlying EigenValues ndim: {my_eigenvalues_ndim}")
    logger.info(f"Underlying EigenValues numpy array: {my_eigenvalues_numpy}")
    logger.info(f"Underlying EigenValues object: {my_eigenvalues_copy}")

    logger.info("Visualizing with Rerun...")
    rr.init("eigenvalues_example", spawn=True)
    datatypes.visualize(my_eigenvalues, entity_path="/EigenValues", label="Original EigenValues")

    # Update the underlying data via the setter -- eigenvalues of a different matrix
    new_w, _ = np.linalg.eigh(np.array([[5.0, 2.0], [2.0, 5.0]], dtype=np.float32))
    my_eigenvalues.data = new_w
    logger.info(f"Updated EigenValues: {my_eigenvalues}")
    datatypes.visualize(
        my_eigenvalues, entity_path="/EigenValues/updated", label="Updated EigenValues"
    )

    # Few checks on the eigenvalues data, returns a boolean
    positive_definite = my_eigenvalues.is_positive_definite()
    logger.info(f"EigenValues are positive definite: {positive_definite}")
    positive_semidefinite = my_eigenvalues.is_positive_semidefinite()
    logger.info(f"EigenValues are positive semidefinite: {positive_semidefinite}")

    # Compute the condition number of the matrix from its eigenvalues
    condition_number = my_eigenvalues.condition_number
    logger.info(f"Condition number of the matrix: {condition_number}")

    # Operate with numpy directly
    spectral_radius = np.max(np.abs(my_eigenvalues))
    logger.info(f"Spectral radius (max |eigenvalue|) via numpy: {spectral_radius}")

    # Serialize to PyArrow and back
    serialization_start_time = time.perf_counter()
    serialized = datatypes.serialize(my_eigenvalues)
    serialization_end_time = time.perf_counter()

    deserialization_start_time = time.perf_counter()
    deserialized = datatypes.deserialize(serialized)["param_0"]
    deserialization_end_time = time.perf_counter()
    logger.info(f"Deserialized EigenValues: {deserialized}")
    logger.info(
        f"Deserialized EigenValues and original EigenValues are equal: {deserialized == my_eigenvalues}"
    )

    logger.info(
        f"Serialization time: {(serialization_end_time - serialization_start_time) * 1000} ms"
    )
    logger.info(
        f"Deserialization time: {(deserialization_end_time - deserialization_start_time) * 1000} ms"
    )


if __name__ == "__main__":
    eigenvalues_example()
