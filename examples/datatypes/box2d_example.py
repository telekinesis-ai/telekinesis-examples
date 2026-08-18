"""Demonstrates the Telekinesis Box2D datatype."""

import time

import numpy as np
from loguru import logger
import rerun as rr

from telekinesis import datatypes

def box2d_example():
    """Demonstrate creation, access, update, translation, scaling, NumPy interop, and serialization."""

    # ======================= Create ============================================
    coords = [1, 2.5, 3, 3]
    box2d = datatypes.Box2D(coords)
    logger.info(f"Original Box2D: {box2d}")

    # ======================= Inspect ===========================================
    logger.info(f"Box2D data: {box2d.data}")
    logger.info(
        f"shape={box2d.shape}, "
        f"width={box2d.width}, "
        f"height={box2d.height}, "
        f"area={box2d.area}, "
        f"center={box2d.center}"
    )

    # ======================= Visualize =========================================
    rr.init("box2d_example", spawn=True)
    datatypes.visualize(box2d, entity_path="/Box2D/my_box2d", label="Original Box2D")

    # ======================= Update ============================================
    updated_coords = [3, 4, 3, 5]
    box2d.data = updated_coords
    logger.info(f"Updated Box2D: {box2d}")
    datatypes.visualize(box2d, entity_path="/Box2D/my_updated_box2d", label="Updated Box2D")

    # ======================= Translate =========================================
    translation = [2, 3]
    translated_box2d = box2d.translate(translation)
    logger.info(f"Translated Box2D: {translated_box2d}")
    datatypes.visualize(
        translated_box2d, entity_path="/Box2D/my_translated_box2d", label="Translated Box2D"
    )

    # ======================= Scale =============================================
    scale_factors = [2, 0.5]
    scaled_box2d = box2d.scale(scale_factors)
    logger.info(f"Scaled Box2D: {scaled_box2d}")
    datatypes.visualize(scaled_box2d, entity_path="/Box2D/my_scaled_box2d", label="Scaled Box2D")

    # ======================= NumPy Interop =====================================
    box2d_xyxy = box2d.convert_box_format(target_format="xyxy")
    scaled_xyxy = scaled_box2d.convert_box_format(target_format="xyxy")
    inter_min = np.maximum(box2d_xyxy[:2], scaled_xyxy.data[:2])
    inter_max = np.minimum(box2d_xyxy[2:], scaled_xyxy.data[2:])
    inter_wh = np.clip(inter_max - inter_min, 0, None)
    intersection = inter_wh[0] * inter_wh[1]
    union = box2d.area + scaled_box2d.area - intersection
    iou = intersection / union if union > 0 else 0.0
    logger.info(f"IoU between box2d and scaled_box2d: {iou}")

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
