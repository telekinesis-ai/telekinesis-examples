"""Demonstrates the Telekinesis Circle datatype."""

import time

import numpy as np
import rerun as rr
from loguru import logger

from telekinesis import datatypes

def circle_example():
    """Demonstrate creation, inspection, operations, visualization, and serialization."""

    # ======================= Create ============================================
    circle = datatypes.Circle(center=[50.0, 60.0], radius=10.0)
    logger.info(f"Original Circle: {circle}")

    # ======================= Inspect ===========================================
    logger.info(f"center={circle.center}")
    logger.info(f"radius={circle.radius}")

    # ======================= Operations =========================================
    # Circle is immutable; translate and scale by constructing a new instance.
    translated_circle = datatypes.Circle(
        center=circle.center + np.array([5.0, 5.0], dtype=np.float32), radius=circle.radius
    )
    logger.info(f"Translated Circle: {translated_circle}")

    scaled_circle = datatypes.Circle(
        center=translated_circle.center, radius=translated_circle.radius * 1.5
    )
    logger.info(f"Scaled Circle: {scaled_circle}")

    # ======================= Visualize =========================================
    rr.init("circle_example", spawn=True)
    datatypes.visualize(circle, entity_path="/circle/original", label="Original Circle")
    datatypes.visualize(scaled_circle, entity_path="/circle/scaled", label="Scaled Circle")

    # ======================= Serialize / Deserialize ===========================
    start = time.perf_counter()
    serialized = datatypes.serialize(scaled_circle)
    serialization_ms = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    deserialized = datatypes.deserialize(serialized)["param_0"]
    deserialization_ms = (time.perf_counter() - start) * 1000

    logger.info(f"Deserialized Circle: {deserialized}")
    logger.info(f"Round-trip successful: {scaled_circle == deserialized}")
    logger.info(f"Serialization time: {serialization_ms:.3f} ms")
    logger.info(f"Deserialization time: {deserialization_ms:.3f} ms")


if __name__ == "__main__":
    circle_example()
