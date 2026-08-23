"""Demonstrates the Telekinesis OrientedBoxes2D datatype."""

import time

import numpy as np
from loguru import logger
import rerun as rr

from telekinesis import datatypes

def oriented_boxes2d_example():
    """Demonstrate creation, access, visualization, update, format conversion, translate/rotate transforms, NumPy corner computation, area ranking, and serialization."""

    # ======================= Create ============================================
    # OrientedBoxes2D format is CXCYWH = [[cx, cy, width, height], ...]
    # + rotation column [yaw_deg]
    box2d_1 = [0.5, 0.5, 0.5, 0.5, 30.0]
    box2d_2 = [1.0, 1.0, 1.0, 1.0, 15.0]
    boxes2d = datatypes.OrientedBoxes2D([box2d_1, box2d_2])

    logger.info(f"Original OrientedBoxes2D: {boxes2d}")

    # ======================= Inspect ===========================================
    logger.info(f"OrientedBoxes2D data: {boxes2d.data}")
    logger.info(
        f"dtype={boxes2d.dtype}, "
        f"ndim={boxes2d.ndim}, "
        f"shape={boxes2d.shape}, "
        f"dimensions={boxes2d.dimensions}, "
        f"areas={boxes2d.areas}, "
        f"centers={boxes2d.centers}, "
        f"rotations={boxes2d.rotations}"
    )

    # ======================= Visualize =========================================
    rr.init("oriented_box2d_example", spawn=True)
    datatypes.visualize(
        boxes2d,
        entity_path="/OrientedBox2D/oriented_boxes2d",
        label=["Oriented Box2D 1", "Oriented Box2D 2"],
    )

    # ======================= Update ============================================
    boxes2d.data = [
        [2.0, 2.0, 1.5, 2.0, 60.0],
        [3.0, 3.0, 2.0, 2.5, 30.0],
    ]
    logger.info(f"Updated OrientedBoxes2D: {boxes2d}")
    datatypes.visualize(
        boxes2d,
        entity_path="/OrientedBoxes2D/updated_oriented_box2d",
        label=["Updated Oriented Box2D 1", "Updated Oriented Box2D 2"],
    )

    # ======================= Alternate Construction =============================
    # Only the center/dimensions portion is reinterpreted; the trailing
    # rotation column passes through unchanged.
    xyxy_coords = [[1.0, 1.5, 3.5, 3.0, 60.0], [2.0, 2.5, 4.5, 4.0, 30.0]]
    boxes_from_xyxy = datatypes.OrientedBoxes2D.from_xyxy(xyxy_coords)
    logger.info(f"OrientedBoxes2D created from xyxy format: {boxes_from_xyxy}")

    xyxy_view = boxes2d.as_xyxy()
    logger.info(f"OrientedBoxes2D converted to xyxy format: {xyxy_view}")

    xywh_coords = [[1.0, 1.5, 2.5, 1.5, 60.0], [2.0, 2.5, 2.5, 1.5, 30.0]]
    boxes_from_xywh = datatypes.OrientedBoxes2D.from_xywh(xywh_coords)
    logger.info(f"OrientedBoxes2D created from xywh format: {boxes_from_xywh}")

    xywh_view = boxes2d.as_xywh()
    logger.info(f"OrientedBoxes2D converted to xywh format: {xywh_view}")

    # ======================= NumPy Interop =====================================
    # Translate and rotate by operating on the underlying NumPy array directly.
    translated_data = boxes2d.data.copy()
    translated_data[:, :2] += [3.0, 3.0]
    translated = datatypes.OrientedBoxes2D(translated_data)
    logger.info(f"Translated centers: {translated.centers} (was {boxes2d.centers})")
    datatypes.visualize(
        translated,
        entity_path="/OrientedBoxes2D/translated_oriented_box2d",
        label=["Translated Oriented Box2D 1", "Translated Oriented Box2D 2"],
    )

    rotated_data = boxes2d.data.copy()
    rotated_data[:, 4] += 15.0
    rotated = datatypes.OrientedBoxes2D(rotated_data)
    logger.info(f"Rotated rotations: {rotated.rotations} (was {boxes2d.rotations})")
    datatypes.visualize(
        rotated,
        entity_path="/OrientedBoxes2D/rotated_oriented_box2d",
        label=["Rotated Oriented Box2D 1", "Rotated Oriented Box2D 2"],
    )

    # ======================= Serialize / Deserialize ===========================
    start = time.perf_counter()
    serialized = datatypes.serialize(boxes2d)
    serialization_ms = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    deserialized = datatypes.deserialize(serialized)["param_0"]
    deserialization_ms = (time.perf_counter() - start) * 1000

    logger.info(f"Deserialized OrientedBoxes2D: {deserialized}")
    logger.info(f"Round-trip successful: {boxes2d == deserialized}")
    logger.info(f"Serialization time: {serialization_ms:.3f} ms")
    logger.info(f"Deserialization time: {deserialization_ms:.3f} ms")


if __name__ == "__main__":
    oriented_boxes2d_example()
