"""
Example script to demonstrate usage of EigenVectors datatype.
"""

import time

import numpy as np
from loguru import logger
import rerun as rr

from telekinesis import datatypes


def eigenvectors_example():
    """
    Example function to demonstrate usage of EigenVectors datatype.
     - Create EigenVectors from the output of np.linalg.eigh
     - Access the underlying eigenvectors through .data (defensive copy)
     - Inspect shape, ndim, dtype and size
     - Visualize the EigenVectors data using Rerun
     - Replace the wrapped payload via the data setter
     - Verify the eigenvector relation m @ v == w * v
     - Loosen the orthonormality check via the atol argument
     - Serialize to PyArrow and back
    """
    # Create EigenVectors from a symmetric matrix decomposition
    input_matrix = np.array([[2.0, 1.0], [1.0, 2.0]], dtype=np.float64)
    w, v = np.linalg.eigh(input_matrix)
    logger.info(f"Input eigenvectors:\n{v}")
    my_eigenvectors = datatypes.EigenVectors(v)
    logger.info(f"Original EigenVectors: {my_eigenvectors}")

    # Access the underlying data
    my_eigenvectors_data = my_eigenvectors.data
    my_eigenvectors_shape = my_eigenvectors.shape
    my_eigenvectors_size = my_eigenvectors.size
    my_eigenvectors_dtype = my_eigenvectors.dtype
    my_eigenvectors_ndim = my_eigenvectors.ndim
    my_eigenvectors_numpy = my_eigenvectors.to_numpy()
    my_eigenvectors_copy = my_eigenvectors.copy()

    logger.info(f"Underlying EigenVectors data:\n{my_eigenvectors_data}")
    logger.info(f"Underlying EigenVectors shape: {my_eigenvectors_shape}")
    logger.info(f"Underlying EigenVectors size: {my_eigenvectors_size}")
    logger.info(f"Underlying EigenVectors dtype: {my_eigenvectors_dtype}")
    logger.info(f"Underlying EigenVectors ndim: {my_eigenvectors_ndim}")
    logger.info(f"Underlying EigenVectors numpy array: {my_eigenvectors_numpy}")
    logger.info(f"Underlying EigenVectors object: {my_eigenvectors_copy}")

    logger.info("Visualizing with Rerun...")
    rr.init("eigenvectors_example", spawn=True)
    datatypes.visualize(my_eigenvectors, entity_path="/EigenVectors", label="Original EigenVectors")

    # Update the underlying data via the eigenvectors of a different matrix
    _, new_v = np.linalg.eigh(np.array([[5.0, 2.0], [2.0, 5.0]], dtype=np.float64))
    my_eigenvectors.data = new_v
    logger.info(f"Updated EigenVectors: {my_eigenvectors}")
    datatypes.visualize(
        my_eigenvectors, entity_path="/EigenVectors/updated", label="Updated EigenVectors"
    )

    # Operate on the underlying data with numpy -- loosen the orthonormality
    # tolerance for noisy inputs
    relaxed = datatypes.EigenVectors(v + 1e-10, atol=1e-6)
    logger.info(f"Noisy input accepted with atol=1e-6: {relaxed}")

    # Verify the eigenvector relation: for each column v_k of the original
    # eigenvectors, input_matrix @ v_k should equal w[k] * v_k
    for k in range(my_eigenvectors.shape[1]):
        v_k = my_eigenvectors_data[:, k]
        lhs = input_matrix @ v_k
        rhs = w[k] * v_k
        logger.info(f"Eigenvector {k} satisfies A @ v == w * v: {np.allclose(lhs, rhs)}")

    # Serialize to PyArrow and back
    serialization_start_time = time.perf_counter()
    serialized = datatypes.serialize(my_eigenvectors)
    serialization_end_time = time.perf_counter()

    deserialization_start_time = time.perf_counter()
    deserialized = datatypes.deserialize(serialized)["param_0"]
    deserialization_end_time = time.perf_counter()
    logger.info(f"Deserialized EigenVectors: {deserialized}")
    logger.info(
        f"Deserialized EigenVectors and original EigenVectors are equal: {deserialized == my_eigenvectors}"
    )

    logger.info(
        f"Serialization time: {(serialization_end_time - serialization_start_time) * 1000} ms"
    )
    logger.info(
        f"Deserialization time: {(deserialization_end_time - deserialization_start_time) * 1000} ms"
    )


if __name__ == "__main__":
    eigenvectors_example()
