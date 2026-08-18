"""Demonstrates the Telekinesis Contour datatype."""

import time

import numpy as np
from loguru import logger
import rerun as rr

from telekinesis import datatypes

def contour_example():
    """Demonstrate creation, inspection, visualization, NumPy-based shifting, and serialization."""

    # ======================= Create ============================================
    contour = datatypes.Contour(
        points=np.array([[118, 84], [134, 79], [150, 86], [139, 115]], dtype=np.int64),
    )
    logger.info(f"Original Contour: {contour}")

    # ======================= Inspect ===========================================
    points = contour.points
    num_points = len(contour)

    logger.info(f"Points: {points}")
    logger.info(f"Number of points: {num_points}")

    # ======================= Visualize =========================================
    rr.init("contour_example", spawn=True)
    datatypes.visualize(contour, entity_path="/Contour")

    # ======================= Shift =============================================
    shifted_points = points + np.array([10, 10], dtype=np.int64)
    shifted_contour = datatypes.Contour(points=shifted_points)
    logger.info(f"Shifted Contour: {shifted_contour}")
    datatypes.visualize(shifted_contour, entity_path="/Contour/shifted")

    # ======================= Serialize / Deserialize ===========================
    start = time.perf_counter()
    serialized = datatypes.serialize(shifted_contour)
    serialization_ms = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    deserialized = datatypes.deserialize(serialized)["param_0"]
    deserialization_ms = (time.perf_counter() - start) * 1000

    logger.info(f"Deserialized Contour: {deserialized}")
    logger.info(f"Round-trip successful: {deserialized == shifted_contour}")
    logger.info(f"Serialization time: {serialization_ms:.3f} ms")
    logger.info(f"Deserialization time: {deserialization_ms:.3f} ms")


if __name__ == "__main__":
    contour_example()
