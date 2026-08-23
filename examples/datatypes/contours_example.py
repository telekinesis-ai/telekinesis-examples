"""Demonstrates the Telekinesis Contours datatype."""

import time

import numpy as np
import rerun as rr
from loguru import logger

from telekinesis import datatypes

def contours_example():
    """Demonstrate creation, inspection, operations, visualization, and serialization."""

    # ======================= Create ============================================
    contour_1 = np.array([[118, 84], [134, 79], [150, 86], [139, 115]], dtype=np.int64)
    contour_2 = np.array([[210, 145], [238, 140], [276, 160], [245, 190]], dtype=np.int64)
    contour_3 = np.array([[322, 212], [335, 211], [332, 225], [320, 218]], dtype=np.int64)
    contours = datatypes.Contours(points=[contour_1, contour_2, contour_3])
    logger.info(f"Created Contours: {contours}")

    empty_points = np.empty((0, 2), dtype=np.int64)
    empty_contours = datatypes.Contours(points=[empty_points])
    logger.info(f"Created Contours with an empty contour: {empty_contours}")

    # ======================= Inspect ===========================================
    logger.info(f"points={contours.points}")
    logger.info(f"length={len(contours)}")

    # ======================= Operations =========================================
    first_contour = contours[0]
    logger.info(f"First contour (index 0): {first_contour}")

    sub_batch = contours[1:]
    logger.info(f"Sub-batch of contours [1:]: {sub_batch}")

    mask = np.array([True, False, True])
    masked_contours = contours[mask]
    logger.info(f"Boolean-masked contours: {masked_contours}")

    # ======================= Visualize =========================================
    rr.init("contours_example", spawn=True)
    datatypes.visualize(contours, entity_path="/contours")

    # ======================= Serialize / Deserialize ===========================
    start = time.perf_counter()
    serialized = datatypes.serialize(contours)
    serialization_ms = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    deserialized = datatypes.deserialize(serialized)["param_0"]
    deserialization_ms = (time.perf_counter() - start) * 1000

    logger.info(f"Deserialized Contours: {deserialized}")
    logger.info(f"Round-trip successful: {contours == deserialized}")
    logger.info(f"Serialization time: {serialization_ms:.3f} ms")
    logger.info(f"Deserialization time: {deserialization_ms:.3f} ms")


if __name__ == "__main__":
    contours_example()
