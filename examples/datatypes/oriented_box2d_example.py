"""Demonstrates the Telekinesis OrientedBox2D datatype."""

import time

import numpy as np
import rerun as rr
from loguru import logger

from telekinesis import datatypes

def oriented_box2d_example():
    """Demonstrate creation, access, visualization, format conversion, translate/scale/rotate, NumPy interop, and serialization."""

    # ======================= Create ============================================
    # OrientedBox2D format is CXCYWH = [cx, cy, width, height] + rotation [yaw_deg]
    box = datatypes.OrientedBox2D([0.5, 0.5, 0.5, 0.5, 30.0])

    logger.info(f"Created OrientedBox2D: {box}")

    # ======================= Inspect ===========================================
    logger.info(f"OrientedBox2D data: {box.data}")
    logger.info(
        f"dtype={box.dtype}, "
        f"ndim={box.ndim}, "
        f"shape={box.shape}, "
        f"dimensions={box.dimensions}, "
        f"area={box.area}, "
        f"center={box.center}, "
        f"rotation={box.rotation}"
    )
    # ======================= Visualize =========================================
    rr.init("oriented_box2d_example", spawn=True)
    datatypes.visualize(
        box, entity_path="/OrientedBox2D/oriented_box2d", label="My Oriented Box2D"
    )

    # ======================= Update ============================================
    box.data = [2.0, 2.0, 1.5, 3.0, 45.0]

    logger.info(f"Updated OrientedBox2D: {box}")
    datatypes.visualize(
        box, entity_path="/OrientedBox2D/updated_oriented_box2d", label="Updated Oriented Box2D"
    )

    # ======================= Alternate Construction =============================
    # Only the center/dimensions portion is reinterpreted; the trailing
    # rotation entry passes through unchanged.
    xyxy_coords = [1.0, 1.5, 3.5, 3.0, 45.0]
    box_from_xyxy = datatypes.OrientedBox2D.from_xyxy(xyxy_coords)
    logger.info(f"OrientedBox2D created from xyxy format: {box_from_xyxy}")

    xyxy_view = box.as_xyxy()
    logger.info(f"OrientedBox2D converted to xyxy format: {xyxy_view}")

    xywh_coords = [1.0, 1.5, 2.5, 1.5, 45.0]
    box_from_xywh = datatypes.OrientedBox2D.from_xywh(xywh_coords)
    logger.info(f"OrientedBox2D created from xywh format: {box_from_xywh}")

    xywh_view = box.as_xywh()
    logger.info(f"OrientedBox2D converted to xywh format: {xywh_view}")

    # ======================= NumPy Interop =====================================
    # Translate, scale, and rotate by operating on the underlying NumPy array directly.
    translated_data = box.data.copy()
    translated_data[:2] += [1.0, 1.0]
    translated_box = datatypes.OrientedBox2D(translated_data)

    logger.info(f"Translated center: {translated_box.center} (was {box.center})")
    datatypes.visualize(
        translated_box,
        entity_path="/OrientedBox2D/translated_oriented_box2d",
        label="Translated Oriented Box2D",
    )

    scaled_data = box.data.copy()
    scaled_data[2:4] *= 1.5
    scaled_box = datatypes.OrientedBox2D(scaled_data)

    logger.info(f"Scaled dimensions: {scaled_box.dimensions} (was {box.dimensions})")
    datatypes.visualize(
        scaled_box, entity_path="/OrientedBox2D/scaled_oriented_box2d", label="Scaled Oriented Box2D"
    )

    rotated_data = box.data.copy()
    rotated_data[4] += 15.0
    rotated_box = datatypes.OrientedBox2D(rotated_data)

    logger.info(f"Rotated rotation: {rotated_box.rotation} (was {box.rotation})")
    datatypes.visualize(
        rotated_box,
        entity_path="/OrientedBox2D/rotated_oriented_box2d",
        label="Rotated Oriented Box2D",
    )

    # ======================= Serialize / Deserialize ===========================
    start = time.perf_counter()
    serialized = datatypes.serialize(box)
    serialization_ms = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    deserialized = datatypes.deserialize(serialized)["param_0"]
    deserialization_ms = (time.perf_counter() - start) * 1000

    logger.info(f"Deserialized OrientedBox2D: {deserialized}")
    logger.info(f"Round-trip successful: {deserialized == box}")
    logger.info(f"Serialization time: {serialization_ms:.3f} ms")
    logger.info(f"Deserialization time: {deserialization_ms:.3f} ms")


if __name__ == "__main__":
    oriented_box2d_example()
