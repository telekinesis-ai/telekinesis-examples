"""Demonstrates the Telekinesis Boxes2D datatype."""

import time

import numpy as np
from loguru import logger
import rerun as rr

from telekinesis import datatypes

def boxes2d_example():
    """Demonstrate creation, access, in-place update, format conversion, translation, scaling, and serialization."""

    # ======================= Create ============================================
    # Boxes2D format is CXCYWH = [[cx, cy, width, height], ...]
    box2d_1 = [[1, 2.5, 3, 3]]
    box2d_2 = [[4, 5, 2, 1]]
    coords = np.concatenate([box2d_1, box2d_2], axis=0)
    boxes2d = datatypes.Boxes2D(coords)
    logger.info(f"Original Boxes2D: {boxes2d}")

    # ======================= Inspect ===========================================
    logger.info(f"Boxes2D data: {boxes2d.data}")
    logger.info(
        f"dtype={boxes2d.dtype}, "
        f"ndim={boxes2d.ndim}, "
        f"shape={boxes2d.shape}, "
        f"dimensions={boxes2d.dimensions}, "
        f"areas={boxes2d.areas}, "
        f"centers={boxes2d.centers}"
    )

    # ======================= Visualize =========================================
    rr.init("boxes2d_example", spawn=True)
    datatypes.visualize(
        boxes2d, entity_path="/Boxes2D/box2d", label=["Original Box2D 1", "Original Box2D 2"]
    )

    # ======================= Update ============================================
    updated_box = [3, 4, 3, 5]
    data = boxes2d.data
    data[1] = updated_box
    boxes2d.data = data
    logger.info(f"Updated Box2D: {boxes2d}")
    datatypes.visualize(
        boxes2d,
        entity_path="/Boxes2D/updated_box2d",
        label=["Original Box2D 1", "Original Box2D 2"],
    )

    # ======================= Alternate Construction =============================
    xyxy_coords = [[1, 1.5, 4, 4.5], [3, 4.5, 5, 5.5]]
    boxes2d_from_xyxy = datatypes.Boxes2D.from_xyxy(xyxy_coords)
    logger.info(f"Boxes2D created from xyxy format: {boxes2d_from_xyxy}")

    xyxy_view = boxes2d.as_xyxy()
    logger.info(f"Boxes2D converted to xyxy format: {xyxy_view}")

    xywh_coords = [[1.0, 1.5, 3.0, 3.0], [3.0, 4.5, 2.0, 1.0]]
    boxes2d_from_xywh = datatypes.Boxes2D.from_xywh(xywh_coords)
    logger.info(f"Boxes2D created from xywh format: {boxes2d_from_xywh}")

    xywh_view = boxes2d.as_xywh()
    logger.info(f"Boxes2D converted to xywh format: {xywh_view}")

    # ======================= NumPy Interop =====================================
    # Translate and scale by operating on the underlying NumPy array directly.
    translation = [2, 3]
    translated_data = boxes2d.data.copy()
    translated_data[:, :2] += translation
    translated_boxes2d = datatypes.Boxes2D(translated_data)
    logger.info(f"Translated Boxes2D: {translated_boxes2d}")
    datatypes.visualize(
        translated_boxes2d,
        entity_path="/Boxes2D/translated_box2d",
        label=["Translated Box2D 1", "Translated Box2D 2"],
    )

    scale_factors = [0.5, 0.5]
    scaled_data = boxes2d.data.copy()
    scaled_data[:, 2:] *= np.asarray(scale_factors, dtype=np.float32)
    scaled_boxes2d = datatypes.Boxes2D(scaled_data)
    logger.info(f"Scaled Boxes2D: {scaled_boxes2d}")
    datatypes.visualize(
        scaled_boxes2d,
        entity_path="/Boxes2D/scaled_box2d",
        label=["Scaled Box2D 1", "Scaled Box2D 2"],
    )

    # ======================= Serialize / Deserialize ===========================
    start = time.perf_counter()
    serialized = datatypes.serialize(boxes2d)
    serialization_ms = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    deserialized = datatypes.deserialize(serialized)["param_0"]
    deserialization_ms = (time.perf_counter() - start) * 1000

    logger.info(f"Deserialized Boxes2D: {deserialized}")
    logger.info(f"Round-trip successful: {boxes2d == deserialized}")
    logger.info(f"Serialization time: {serialization_ms:.3f} ms")
    logger.info(f"Deserialization time: {deserialization_ms:.3f} ms")


if __name__ == "__main__":
    boxes2d_example()
