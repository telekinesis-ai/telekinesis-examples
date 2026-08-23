"""Demonstrates the Telekinesis Box2D datatype."""

import time

import numpy as np
import rerun as rr
from loguru import logger

from telekinesis import datatypes

def box2d_example():
    """Demonstrate creation, inspection, operations, visualization, and serialization."""

    # ======================= Create ============================================
    # Box2D format is CXCYWH = [cx, cy, width, height]
    coords = [1, 2.5, 3, 3]
    box2d = datatypes.Box2D(coords)
    logger.info(f"Original Box2D: {box2d}")

    xyxy_coords = [1.0, 1.5, 4.0, 4.5]
    box2d_from_xyxy = datatypes.Box2D.from_xyxy(xyxy_coords)
    logger.info(f"Box2D created from xyxy format: {box2d_from_xyxy}")

    xywh_coords = [1.0, 1.5, 3.0, 3.0]
    box2d_from_xywh = datatypes.Box2D.from_xywh(xywh_coords)
    logger.info(f"Box2D created from xywh format: {box2d_from_xywh}")

    # ======================= Inspect ===========================================
    logger.info(f"data={box2d.data}")
    logger.info(f"dtype={box2d.dtype}")
    logger.info(f"ndim={box2d.ndim}")
    logger.info(f"shape={box2d.shape}")
    logger.info(f"size={box2d.size}")
    logger.info(f"dimensions={box2d.dimensions}")
    logger.info(f"area={box2d.area}")
    logger.info(f"center={box2d.center}")

    # ======================= Operations =========================================
    updated_coords = [3, 4, 3, 5]
    box2d.data = updated_coords
    logger.info(f"Updated Box2D: {box2d}")

    xyxy_view = box2d.as_xyxy()
    logger.info(f"Box2D converted to xyxy format: {xyxy_view}")

    xywh_view = box2d.as_xywh()
    logger.info(f"Box2D converted to xywh format: {xywh_view}")

    box2d_copy = box2d.copy()
    logger.info(f"Copied Box2D: {box2d_copy}")

    # Returns the internal data as a NumPy array. If copy=True, returns a copy; otherwise, returns a view.
    box2d_numpy = box2d.to_numpy(copy=False)
    logger.info(f"NumPy Box2D:\n{box2d_numpy}")

    # __array__ is implemented, so Box2D works directly with NumPy functions.
    logger.info(f"np.asarray(box2d)={np.asarray(box2d)}")
    logger.info(f"np.sum(box2d)={np.sum(box2d)}")

    # Translate and scale by operating on the underlying NumPy array directly.
    translation = [2, 3]
    translated_data = box2d.data.copy()
    translated_data[:2] += translation
    translated_box2d = datatypes.Box2D(translated_data)
    logger.info(f"Translated Box2D: {translated_box2d}")

    scale_factors = [2, 0.5]
    scaled_data = box2d.data.copy()
    scaled_data[2:] *= np.asarray(scale_factors, dtype=np.float32)
    scaled_box2d = datatypes.Box2D(scaled_data)
    logger.info(f"Scaled Box2D: {scaled_box2d}")

    # ======================= Visualize =========================================
    rr.init("box2d_example", spawn=True)
    datatypes.visualize(box2d, entity_path="/box2d/updated", label="Updated Box2D")
    datatypes.visualize(
        translated_box2d, entity_path="/box2d/translated", label="Translated Box2D"
    )
    datatypes.visualize(scaled_box2d, entity_path="/box2d/scaled", label="Scaled Box2D")

    # ======================= Serialize / Deserialize ===========================
    start = time.perf_counter()
    serialized = datatypes.serialize(box2d)
    serialization_ms = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    deserialized = datatypes.deserialize(serialized)["param_0"]
    deserialization_ms = (time.perf_counter() - start) * 1000

    logger.info(f"Deserialized Box2D: {deserialized}")
    logger.info(f"Round-trip successful: {box2d == deserialized}")
    logger.info(f"Serialization time: {serialization_ms:.3f} ms")
    logger.info(f"Deserialization time: {deserialization_ms:.3f} ms")


if __name__ == "__main__":
    box2d_example()
