"""Demonstrates the Telekinesis OrientedBoxes2D datatype."""

import time

import numpy as np
from loguru import logger
import rerun as rr

from telekinesis import datatypes

def oriented_boxes2d_example():
    """Demonstrate creation, access, visualization, update, translate/rotate transforms, NumPy corner computation, area ranking, and serialization."""

    # ======================= Create ============================================
    box2d_1 = [0.5, 0.5, 0.5, 0.5, 0.5]
    box2d_2 = [1.0, 1.0, 1.0, 1.0, 0.25]
    boxes2d = datatypes.OrientedBoxes2D([box2d_1, box2d_2])

    logger.info(f"Original OrientedBoxes2D: {boxes2d}")

    # ======================= Inspect ===========================================
    data = boxes2d.data
    shape = boxes2d.shape
    dtype = boxes2d.dtype
    ndim = boxes2d.ndim
    numpy_boxes2d = boxes2d.to_numpy()
    center = boxes2d.center
    area = boxes2d.area
    width = boxes2d.width
    height = boxes2d.height
    theta = boxes2d.theta

    logger.info(f"shape={shape}, dtype={dtype}, ndim={ndim}")
    logger.info(f"Underlying data: {data}")
    logger.info(f"NumPy array: {numpy_boxes2d}")
    logger.info(
        f"center={center}, area={area}, width={width}, height={height}, theta={theta}"
    )

    # ======================= Visualize =========================================
    rr.init("oriented_box2d_example", spawn=True)
    datatypes.visualize(
        boxes2d,
        entity_path="/OrientedBox2D/my_oriented_boxes2d",
        label=["My Oriented Box2D 1", "My Oriented Box2D 2"],
    )

    # ======================= Update ============================================
    boxes2d.data = [
        [2.0, 2.0, 1.5, 1.0, 1.0],
        [3.0, 3.0, 2.0, 1.5, 0.5],
    ]
    logger.info(f"Updated OrientedBoxes2D: {boxes2d}")
    datatypes.visualize(
        boxes2d,
        entity_path="/OrientedBoxes2D/my_updated_oriented_box2d",
        label=["Updated Oriented Box2D 1", "Updated Oriented Box2D 2"],
    )

    # ======================= Translate =========================================
    translated = boxes2d.translate([3.0, 3.0])
    logger.info(f"Translated center: {translated.center} (was {boxes2d.center})")
    datatypes.visualize(
        translated,
        entity_path="/OrientedBoxes2D/my_translated_oriented_box2d",
        label=["Translated Oriented Box2D 1", "Translated Oriented Box2D 2"],
    )

    # ======================= Rotate ============================================
    rotated = boxes2d.rotate(0.25)
    logger.info(f"Rotated theta: {rotated.theta} (was {boxes2d.theta})")
    datatypes.visualize(
        rotated,
        entity_path="/OrientedBoxes2D/my_rotated_oriented_box2d",
        label=["Rotated Oriented Box2D 1", "Rotated Oriented Box2D 2"],
    )

    # ======================= NumPy Interop =====================================
    data = np.asarray(rotated)
    centers = data[:, :2]
    half_extents = data[:, 2:4] / 2
    angles = data[:, 4]

    corner_signs = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=np.float32)
    local_corners = corner_signs[None, :, :] * half_extents[:, None, :]

    cos_t, sin_t = np.cos(angles), np.sin(angles)
    rotation_matrices = np.stack(
        [np.stack([cos_t, -sin_t], axis=-1), np.stack([sin_t, cos_t], axis=-1)], axis=1
    )

    corners = local_corners @ rotation_matrices.transpose(0, 2, 1) + centers[:, None, :]
    logger.info(f"Corners per box, world space, shape {corners.shape}:\n{corners}")

    edge_w = np.linalg.norm(corners[:, 1] - corners[:, 0], axis=-1)
    edge_h = np.linalg.norm(corners[:, 2] - corners[:, 1], axis=-1)
    logger.info(f"Area from numpy corners: {edge_w * edge_h} (matches .area: {rotated.area})")

    # ======================= Rank by Area ======================================
    order = np.argsort(-rotated.area)
    largest_first = datatypes.OrientedBoxes2D(rotated.data[order])
    logger.info(
        f"Boxes ranked by area (largest first): {largest_first.area} (order: {order.tolist()})"
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
