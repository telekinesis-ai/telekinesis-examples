"""Demonstrates the Telekinesis Box2D datatype."""

import time

import numpy as np
from loguru import logger
import rerun as rr

from telekinesis import datatypes

def box2d_example():
    """Demonstrate creation, access, update, format conversion, translation, scaling, NumPy interop, and serialization."""

    # ======================= Create ============================================
    # Box2D format is CXCYWH = [cx, cy, width, height]
    coords = [1, 2.5, 3, 3]
    box2d = datatypes.Box2D(coords)
    logger.info(f"Original Box2D: {box2d}")

    # ======================= Inspect ===========================================
    logger.info(f"Box2D data: {box2d.data}")
    logger.info(
        f"dtype={box2d.dtype}, "
        f"ndim={box2d.ndim}, "
        f"shape={box2d.shape}, "
        f"dimensions={box2d.dimensions}, "
        f"area={box2d.area}, "
        f"center={box2d.center}"
    )

    # ======================= Visualize =========================================
    rr.init("box2d_example", spawn=True)
    datatypes.visualize(box2d, entity_path="/Box2D/box2d", label="Original Box2D")

    # ======================= Update ============================================
    updated_coords = [3, 4, 3, 5]
    box2d.data = updated_coords
    logger.info(f"Updated Box2D: {box2d}")
    datatypes.visualize(box2d, entity_path="/Box2D/updated_box2d", label="Updated Box2D")

    # ======================= Alternate Construction =============================
    xyxy_coords = [1.0, 1.5, 4.0, 4.5]
    box2d_from_xyxy = datatypes.Box2D.from_xyxy(xyxy_coords)
    logger.info(f"Box2D created from xyxy format: {box2d_from_xyxy}")

    xyxy_view = box2d.as_xyxy()
    logger.info(f"Box2D converted to xyxy format: {xyxy_view}")

    xywh_coords = [1.0, 1.5, 3.0, 3.0]
    box2d_from_xywh = datatypes.Box2D.from_xywh(xywh_coords)
    logger.info(f"Box2D created from xywh format: {box2d_from_xywh}")

    xywh_view = box2d.as_xywh()
    logger.info(f"Box2D converted to xywh format: {xywh_view}")

    # ======================= Other Methods =====================================
    box2d_copy = box2d.copy()
    logger.info(f"Copied Box2D: {box2d_copy}")

    # Returns the internal data as a NumPy array. If copy=True, returns a copy; otherwise, returns a view.
    box2d_numpy = box2d.to_numpy(copy=False)
    logger.info(f"NumPy Box2D:\n{box2d_numpy}")

    # ======================= NumPy Interop =====================================
    # Translate and scale by operating on the underlying NumPy array directly.
    translation = [2, 3]
    translated_data = box2d.data.copy()
    translated_data[:2] += translation
    translated_box2d = datatypes.Box2D(translated_data)
    logger.info(f"Translated Box2D: {translated_box2d}")
    datatypes.visualize(
        translated_box2d, entity_path="/Box2D/translated_box2d", label="Translated Box2D"
    )

    scale_factors = [2, 0.5]
    scaled_data = box2d.data.copy()
    scaled_data[2:] *= np.asarray(scale_factors, dtype=np.float32)
    scaled_box2d = datatypes.Box2D(scaled_data)
    logger.info(f"Scaled Box2D: {scaled_box2d}")
    datatypes.visualize(scaled_box2d, entity_path="/Box2D/scaled_box2d", label="Scaled Box2D")

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
