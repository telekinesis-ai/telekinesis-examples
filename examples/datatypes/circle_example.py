"""Demonstrates the Telekinesis Circle datatype."""

import time

import numpy as np
import rerun as rr
from loguru import logger

from telekinesis import datatypes

def circle_example():
    """Demonstrate creation, access, translation and scaling of the immutable Circle, visualization, and serialization."""

    # ======================= Create ============================================
    circle = datatypes.Circle(center=[50.0, 60.0], radius=10.0)
    logger.info(f"Original Circle: {circle}")

    # ======================= Inspect ===========================================
    logger.info(f"center={circle.center}, radius={circle.radius}")

    # ======================= Translate / Scale =================================
    # `translate()`/`scale()` were removed; build new `Circle`s directly.
    translated = datatypes.Circle(
        center=circle.center + np.array([5.0, 5.0], dtype=np.float32), radius=circle.radius
    )
    scaled = datatypes.Circle(center=translated.center, radius=translated.radius * 1.5)
    logger.info(f"Translated Circle: {translated}")
    logger.info(f"Scaled Circle: {scaled}")

    # ======================= Visualize =========================================
    rr.init("circle_example", spawn=True)
    datatypes.visualize(circle, entity_path="/Circle", label="My Circle")
    datatypes.visualize(scaled, entity_path="/Circle/updated", label="Updated Circle")

    # ======================= Serialize / Deserialize ===========================
    start = time.perf_counter()
    serialized = datatypes.serialize(scaled)
    serialization_ms = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    deserialized = datatypes.deserialize(serialized)["param_0"]
    deserialization_ms = (time.perf_counter() - start) * 1000

    logger.info(f"Deserialized Circle: {deserialized}")
    logger.info(f"Round-trip successful: {scaled == deserialized}")
    logger.info(f"Serialization time: {serialization_ms:.3f} ms")
    logger.info(f"Deserialization time: {deserialization_ms:.3f} ms")


if __name__ == "__main__":
    circle_example()
