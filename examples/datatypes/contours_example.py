"""Demonstrates the Telekinesis Contours datatype."""

import time

import rerun as rr
import numpy as np
from loguru import logger

from telekinesis import datatypes

def contours_example():
    """Demonstrate creation, inspection, visualization, indexing, empty contours, and serialization."""

    # ======================= Create ============================================
    contour_1 = np.array([[118, 84], [134, 79], [150, 86], [139, 115]], dtype=np.int64)
    contour_2 = np.array([[210, 145], [238, 140], [276, 160], [245, 190]], dtype=np.int64)
    contour_3 = np.array([[322, 212], [335, 211], [332, 225], [320, 218]], dtype=np.int64)
    contours = datatypes.Contours(points=[contour_1, contour_2, contour_3])
    logger.info(f"Original Contours: {contours}")

    # ======================= Inspect ===========================================
    logger.info(f"Points: {contours.points}")
    logger.info(f"First contour points: {contours.points[0]}")
    logger.info(f"Number of contours: {len(contours)}")

    # ======================= Visualize =========================================
    rr.init("contours_example", spawn=True)
    datatypes.visualize(contours, entity_path="/Contours")

    # ======================= Index =============================================
    logger.info(f"First contour: {contours[0]}")
    logger.info(f"Sub-batch of contours [1:]: {contours[1:]}")

    # ======================= Empty Contours ====================================
    empty_points = np.empty((0, 2), dtype=np.int64)
    empty_contours = datatypes.Contours(points=[empty_points])
    logger.info(f"Empty Contours: {empty_contours}")

    # ======================= Serialize / Deserialize ===========================
    start = time.perf_counter()
    serialized = datatypes.serialize(contours)
    serialization_ms = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    deserialized = datatypes.deserialize(serialized)["param_0"]
    deserialization_ms = (time.perf_counter() - start) * 1000

    logger.info(f"Deserialized Contours: {deserialized}")
    logger.info(f"Round-trip successful: {deserialized == contours}")
    logger.info(f"Serialization time: {serialization_ms:.3f} ms")
    logger.info(f"Deserialization time: {deserialization_ms:.3f} ms")


if __name__ == "__main__":
    contours_example()
