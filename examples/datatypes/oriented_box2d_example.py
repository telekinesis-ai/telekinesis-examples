"""Demonstrates the Telekinesis OrientedBox2D datatype."""

import time

import numpy as np
from loguru import logger
import rerun as rr

from telekinesis import datatypes

def oriented_box2d_example():
    """Demonstrate creation, access, visualization, translate/scale/rotate, NumPy interop, and serialization."""

    # ======================= Create ============================================
    box = datatypes.OrientedBox2D([0.5, 0.5, 0.5, 0.5, 0.5])

    logger.info(f"Created OrientedBox2D: {box}")

    # ======================= Visualize =========================================
    rr.init("oriented_box2d_example", spawn=True)
    datatypes.visualize(
        box, entity_path="/OrientedBox2D/my_oriented_box2d", label="My Oriented Box2D"
    )

    # ======================= Inspect ===========================================
    logger.info(f"shape={box.shape}, dtype={box.dtype}, ndim={box.ndim}")
    logger.info(f"NumPy array: {box.to_numpy()}")
    logger.info(
        f"center={box.center}, area={box.area}, width={box.width}, "
        f"height={box.height}, theta={box.theta}"
    )

    # ======================= Update ============================================
    box.data = [2.0, 2.0, 1.5, 1.0, 1.0]

    logger.info(f"Updated OrientedBox2D: {box}")
    datatypes.visualize(
        box, entity_path="/OrientedBox2D/my_updated_oriented_box2d", label="Updated Oriented Box2D"
    )

    # ======================= Translate =========================================
    translated_box = box.translate([1.0, 1.0])

    logger.info(f"Translated center: {translated_box.center} (was {box.center})")
    datatypes.visualize(
        translated_box,
        entity_path="/OrientedBox2D/my_translated_oriented_box2d",
        label="Translated Oriented Box2D",
    )

    # ======================= Scale =============================================
    scaled_box = box.scale(1.5)

    logger.info(
        f"Scaled width and height: {scaled_box.width} x {scaled_box.height} "
        f"(was {box.width} x {box.height})"
    )
    datatypes.visualize(
        scaled_box, entity_path="/OrientedBox2D/my_scaled_oriented_box2d", label="Scaled Oriented Box2D"
    )

    # ======================= Rotate ============================================
    rotated_box = box.rotate(0.25)

    logger.info(f"Rotated theta: {rotated_box.theta} (was {box.theta})")
    datatypes.visualize(
        rotated_box,
        entity_path="/OrientedBox2D/my_rotated_oriented_box2d",
        label="Rotated Oriented Box2D",
    )

    # ======================= NumPy Interop =====================================
    cx, cy, w, h, theta = np.asarray(rotated_box)
    local_corners = np.array(
        [[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]], dtype=np.float32
    )
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    rotation_matrix = np.array([[cos_t, -sin_t], [sin_t, cos_t]], dtype=np.float32)
    corners = local_corners @ rotation_matrix.T + np.array([cx, cy], dtype=np.float32)

    logger.info(f"Corners (world space, [x, y] per row):\n{corners}")

    edge_w = np.linalg.norm(corners[1] - corners[0])
    edge_h = np.linalg.norm(corners[2] - corners[1])

    logger.info(f"Area from corners: {edge_w * edge_h} (matches .area: {rotated_box.area})")

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
