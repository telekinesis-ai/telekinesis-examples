"""Demonstrates the Telekinesis Circles datatype."""

import time

import numpy as np
from loguru import logger
import rerun as rr

from telekinesis import datatypes

def circles_example():
    """Demonstrate creation, access, indexing, translation, scaling, and serialization of a Circles batch."""

    # ======================= Create ============================================
    circles = datatypes.Circles(
        centers=np.array([[50.0, 60.0], [120.0, 80.0], [200.0, 150.0]], dtype=np.float32),
        radii=np.array([10.0, 15.5, 7.25], dtype=np.float32),
    )
    logger.info(f"Original Circles: {circles}")

    # ======================= Inspect ===========================================
    logger.info(f"centers={circles.centers}, radii={circles.radii}")

    # ======================= Visualize =========================================
    rr.init("circles_example", spawn=True)
    datatypes.visualize(
        circles,
        entity_path="/Circles",
        label=["My Circle 1", "My Circle 2", "My Circle 3"],
    )

    # ======================= Index =============================================
    first = circles[0]
    sub_batch = circles[1:]
    logger.info(f"First circle: center={first.center}, radius={first.radius}")
    logger.info(f"Sub-batch [1:]: centers={sub_batch.centers}, radii={sub_batch.radii}")

    # ======================= Translate / Scale =================================
    circles = circles.translate([[5.0, 5.0], [10.0, 0.0], [-5.0, -5.0]]).scale(1.5)
    logger.info(f"Updated Circles centers data: {circles.centers}")
    logger.info(f"Updated Circles radii data: {circles.radii}")
    datatypes.visualize(
        circles,
        entity_path="/Circles/updated",
        label=["Updated Circle 1", "Updated Circle 2", "Updated Circle 3"],
    )

    # ======================= Empty =============================================
    empty_centers = np.empty((0, 2), dtype=np.float32)
    empty_radii = np.empty((0,), dtype=np.float32)
    empty = datatypes.Circles(centers=empty_centers, radii=empty_radii)
    logger.info(f"Empty Circles: {empty}")

    # ======================= Serialize / Deserialize ===========================
    start = time.perf_counter()
    serialized = datatypes.serialize(circles)
    serialization_ms = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    deserialized = datatypes.deserialize(serialized)["param_0"]
    deserialization_ms = (time.perf_counter() - start) * 1000

    logger.info(f"Deserialized Circles: {deserialized}")
    logger.info(f"Round-trip successful: {circles == deserialized}")
    logger.info(f"Serialization time: {serialization_ms:.3f} ms")
    logger.info(f"Deserialization time: {deserialization_ms:.3f} ms")


if __name__ == "__main__":
    circles_example()
