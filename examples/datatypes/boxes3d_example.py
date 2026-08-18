"""Demonstrates the Telekinesis Boxes3D datatype."""

import time

import numpy as np
from loguru import logger
import rerun as rr

from telekinesis import datatypes

def boxes3d_example():
    """Demonstrate creation, access, in-place update, translation, scaling, and serialization."""

    # ======================= Create ============================================
    box3d_1 = [[0, 0, 0, 1, 1, 1]]
    box3d_2 = [[2, 2, 2, 3, 3, 3]]
    coords = np.concatenate([box3d_1, box3d_2], axis=0)
    boxes3d = datatypes.Boxes3D(coords)
    logger.info(f"Original Boxes3D: {boxes3d}")

    # ======================= Inspect ===========================================
    logger.info(f"Boxes3D data: {boxes3d.data}")
    logger.info(
        f"shape={boxes3d.shape}, "
        f"width={boxes3d.width}, "
        f"height={boxes3d.height}, "
        f"depth={boxes3d.depth}, "
        f"volume={boxes3d.volume}, "
        f"center={boxes3d.center}"
    )

    # ======================= Visualize =========================================
    rr.init("boxes3d_example", spawn=True)
    datatypes.visualize(
        boxes3d, entity_path="/Boxes3D/my_box3d", label=["Original Box3D 1", "Original Box3D 2"]
    )

    # ======================= Update ============================================
    updated_box = [3, 3, 3, 1, 1, 1]
    data = boxes3d.data
    data[1] = updated_box
    boxes3d.data = data
    logger.info(f"Updated Box3D: {boxes3d}")
    datatypes.visualize(
        boxes3d,
        entity_path="/Boxes3D/my_updated_box3d",
        label=["Original Box3D 1", "Original Box3D 2"],
    )

    # ======================= Translate =========================================
    translation = [2, 3, 1]
    translated_boxes3d = boxes3d.translate(translation)
    logger.info(f"Translated Boxes3D: {translated_boxes3d}")
    datatypes.visualize(
        translated_boxes3d,
        entity_path="/Boxes3D/my_translated_box3d",
        label=["Translated Box3D 1", "Translated Box3D 2"],
    )

    # ======================= Scale =============================================
    scale_factors = [0.5, 0.5, 0.5]
    scaled_boxes3d = boxes3d.scale(scale_factors)
    logger.info(f"Scaled Boxes3D: {scaled_boxes3d}")
    datatypes.visualize(
        scaled_boxes3d,
        entity_path="/Boxes3D/my_scaled_box3d",
        label=["Scaled Box3D 1", "Scaled Box3D 2"],
    )

    # ======================= Serialize / Deserialize ===========================
    start = time.perf_counter()
    serialized = datatypes.serialize(boxes3d)
    serialization_ms = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    deserialized = datatypes.deserialize(serialized)["param_0"]
    deserialization_ms = (time.perf_counter() - start) * 1000

    logger.info(f"Deserialized Boxes3D: {deserialized}")
    logger.info(f"Round-trip successful: {boxes3d == deserialized}")
    logger.info(f"Serialization time: {serialization_ms:.3f} ms")
    logger.info(f"Deserialization time: {deserialization_ms:.3f} ms")


if __name__ == "__main__":
    boxes3d_example()
