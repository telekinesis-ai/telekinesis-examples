"""Demonstrates the Telekinesis Circle datatype."""

import time

import numpy as np
from loguru import logger
import rerun as rr

from telekinesis import datatypes

def circle_example():
    """Demonstrate creation, access, translation and scaling of the immutable Circle, visualization, and serialization."""

    # ======================= Create ============================================
    circle = datatypes.Circle(center=[50.0, 60.0], radius=10.0)
    logger.info(f"Original Circle: {circle}")

    # ======================= Inspect ===========================================
    logger.info(f"center={circle.center}, radius={circle.radius}")

    # ======================= Translate / Scale =================================
    translated = circle.translate([5.0, 5.0])
    scaled = translated.scale(1.5)
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
