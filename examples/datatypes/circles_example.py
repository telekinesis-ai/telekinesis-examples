"""Demonstrates the Telekinesis Circles datatype."""

import time

import numpy as np
import rerun as rr
from loguru import logger

from telekinesis import datatypes

def circles_example():
    """Demonstrate creation, inspection, operations, visualization, and serialization."""

    # ======================= Create ============================================
    circles = datatypes.Circles(
        centers=np.array([[50.0, 60.0], [120.0, 80.0], [200.0, 150.0]], dtype=np.float32),
        radii=np.array([10.0, 15.5, 7.25], dtype=np.float32),
    )
    logger.info(f"Original Circles: {circles}")

    circles_from_dict = datatypes.Circles.coerce(
        {"centers": [[0.0, 0.0], [10.0, 10.0]], "radii": [1.0, 2.0]}
    )
    logger.info(f"Circles coerced from dict: {circles_from_dict}")

    # ======================= Inspect ===========================================
    logger.info(f"centers={circles.centers}")
    logger.info(f"radii={circles.radii}")

    # ======================= Operations =========================================
    logger.info(f"Number of circles: {len(circles)}")

    first_circle = circles[0]
    sub_batch = circles[1:]
    logger.info(f"First circle: {first_circle}")
    logger.info(f"Sub-batch [1:]: {sub_batch}")

    # Circles is immutable; translate and scale by constructing a new instance.
    offsets = np.array([[5.0, 5.0], [10.0, 0.0], [-5.0, -5.0]], dtype=np.float32)
    translated_circles = datatypes.Circles(centers=circles.centers + offsets, radii=circles.radii)
    logger.info(f"Translated Circles: {translated_circles}")

    scaled_circles = datatypes.Circles(
        centers=translated_circles.centers, radii=translated_circles.radii * 1.5
    )
    logger.info(f"Scaled Circles: {scaled_circles}")

    # ======================= Visualize =========================================
    rr.init("circles_example", spawn=True)
    datatypes.visualize(
        circles,
        entity_path="/circles/original",
        label=["Original Circle 1", "Original Circle 2", "Original Circle 3"],
    )
    datatypes.visualize(
        scaled_circles,
        entity_path="/circles/scaled",
        label=["Scaled Circle 1", "Scaled Circle 2", "Scaled Circle 3"],
    )

    # ======================= Serialize / Deserialize ===========================
    start = time.perf_counter()
    serialized = datatypes.serialize(scaled_circles)
    serialization_ms = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    deserialized = datatypes.deserialize(serialized)["param_0"]
    deserialization_ms = (time.perf_counter() - start) * 1000

    logger.info(f"Deserialized Circles: {deserialized}")
    logger.info(f"Round-trip successful: {scaled_circles == deserialized}")
    logger.info(f"Serialization time: {serialization_ms:.3f} ms")
    logger.info(f"Deserialization time: {deserialization_ms:.3f} ms")


if __name__ == "__main__":
    circles_example()
