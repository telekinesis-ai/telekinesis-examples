"""Demonstrates the Telekinesis Boxes2D datatype."""

import time

import numpy as np
from loguru import logger
import rerun as rr

from telekinesis import datatypes

def boxes2d_example():
    """Demonstrate creation, access, in-place update, translation, scaling, and serialization."""

    # ======================= Create ============================================
    box2d_1 = [[1, 2.5, 3, 3]]
    box2d_2 = [[4, 5, 2, 1]]
    coords = np.concatenate([box2d_1, box2d_2], axis=0)
    boxes2d = datatypes.Boxes2D(coords)
    logger.info(f"Original Boxes2D: {boxes2d}")

    # ======================= Inspect ===========================================
    logger.info(f"Boxes2D data: {boxes2d.data}")
    logger.info(
        f"shape={boxes2d.shape}, "
        f"width={boxes2d.width}, "
        f"height={boxes2d.height}, "
        f"area={boxes2d.area}, "
        f"center={boxes2d.center}"
    )

    # ======================= Visualize =========================================
    rr.init("boxes2d_example", spawn=True)
    datatypes.visualize(
        boxes2d, entity_path="/Boxes2D/my_box2d", label=["Original Box2D 1", "Original Box2D 2"]
    )

    # ======================= Update ============================================
    updated_box = [3, 4, 3, 5]
    data = boxes2d.data
    data[1] = updated_box
    boxes2d.data = data
    logger.info(f"Updated Box2D: {boxes2d}")
    datatypes.visualize(
        boxes2d,
        entity_path="/Boxes2D/my_updated_box2d",
        label=["Original Box2D 1", "Original Box2D 2"],
    )

    # ======================= Translate =========================================
    translation = [2, 3]
    translated_boxes2d = boxes2d.translate(translation)
    logger.info(f"Translated Boxes2D: {translated_boxes2d}")
    datatypes.visualize(
        translated_boxes2d,
        entity_path="/Boxes2D/my_translated_box2d",
        label=["Translated Box2D 1", "Translated Box2D 2"],
    )

    # ======================= Scale =============================================
    scale_factors = [0.5, 0.5]
    scaled_boxes2d = boxes2d.scale(scale_factors)
    logger.info(f"Scaled Boxes2D: {scaled_boxes2d}")
    datatypes.visualize(
        scaled_boxes2d,
        entity_path="/Boxes2D/my_scaled_box2d",
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
