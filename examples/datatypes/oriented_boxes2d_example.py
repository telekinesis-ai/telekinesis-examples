"""Demonstrates the Telekinesis OrientedBoxes2D datatype."""

import time

import numpy as np
import rerun as rr
from loguru import logger

from telekinesis import datatypes

def oriented_boxes2d_example():
    """Demonstrate creation, inspection, operations, visualization, and serialization."""

    # ======================= Create ============================================
    # OrientedBoxes2D format is CXCYWH = [[cx, cy, width, height], ...]
    # + rotation column [yaw_deg]
    oriented_box2d_1 = [0.5, 0.5, 0.5, 0.5, 30.0]
    oriented_box2d_2 = [1.0, 1.0, 1.0, 1.0, 15.0]
    oriented_boxes2d = datatypes.OrientedBoxes2D([oriented_box2d_1, oriented_box2d_2])
    logger.info(f"Created OrientedBoxes2D: {oriented_boxes2d}")

    xyxy_coords = [[1.0, 1.5, 3.5, 3.0, 60.0], [2.0, 2.5, 4.5, 4.0, 30.0]]
    oriented_boxes2d_from_xyxy = datatypes.OrientedBoxes2D.from_xyxy(xyxy_coords)
    logger.info(f"OrientedBoxes2D created from xyxy format: {oriented_boxes2d_from_xyxy}")

    xywh_coords = [[1.0, 1.5, 2.5, 1.5, 60.0], [2.0, 2.5, 2.5, 1.5, 30.0]]
    oriented_boxes2d_from_xywh = datatypes.OrientedBoxes2D.from_xywh(xywh_coords)
    logger.info(f"OrientedBoxes2D created from xywh format: {oriented_boxes2d_from_xywh}")

    # ======================= Inspect ===========================================
    logger.info(f"data={oriented_boxes2d.data}")
    logger.info(f"dtype={oriented_boxes2d.dtype}")
    logger.info(f"ndim={oriented_boxes2d.ndim}")
    logger.info(f"shape={oriented_boxes2d.shape}")
    logger.info(f"size={oriented_boxes2d.size}")
    logger.info(f"length={len(oriented_boxes2d)}")
    logger.info(f"centers={oriented_boxes2d.centers}")
    logger.info(f"dimensions={oriented_boxes2d.dimensions}")
    logger.info(f"areas={oriented_boxes2d.areas}")
    logger.info(f"rotations={oriented_boxes2d.rotations}")

    # ======================= Operations =========================================
    updated_data = [
        [2.0, 2.0, 1.5, 2.0, 60.0],
        [3.0, 3.0, 2.0, 2.5, 30.0],
    ]
    oriented_boxes2d.data = updated_data
    logger.info(f"Updated OrientedBoxes2D: {oriented_boxes2d}")

    xyxy_view = oriented_boxes2d.as_xyxy()
    logger.info(f"OrientedBoxes2D converted to xyxy format: {xyxy_view}")

    xywh_view = oriented_boxes2d.as_xywh()
    logger.info(f"OrientedBoxes2D converted to xywh format: {xywh_view}")

    first_oriented_box2d = oriented_boxes2d[0]
    logger.info(f"First OrientedBox2D (index 0): {first_oriented_box2d}")

    sub_batch = oriented_boxes2d[1:]
    logger.info(f"Sub-batch of OrientedBoxes2D [1:]: {sub_batch}")

    oriented_boxes2d_copy = oriented_boxes2d.copy()
    logger.info(f"Copied OrientedBoxes2D: {oriented_boxes2d_copy}")

    # Returns the internal data as a NumPy array. If copy=True, returns a copy; otherwise, returns a view.
    oriented_boxes2d_numpy = oriented_boxes2d.to_numpy(copy=False)
    logger.info(f"NumPy OrientedBoxes2D:\n{oriented_boxes2d_numpy}")

    # Translate and rotate by operating on the underlying NumPy array directly.
    translation = [1.0, 1.0]
    translated_data = oriented_boxes2d.data.copy()
    translated_data[:, :2] += translation
    translated_oriented_boxes2d = datatypes.OrientedBoxes2D(translated_data)
    logger.info(f"Translated OrientedBoxes2D: {translated_oriented_boxes2d}")

    rotation_delta_deg = 15.0
    rotated_data = oriented_boxes2d.data.copy()
    rotated_data[:, 4] += rotation_delta_deg
    rotated_oriented_boxes2d = datatypes.OrientedBoxes2D(rotated_data)
    logger.info(f"Rotated OrientedBoxes2D: {rotated_oriented_boxes2d}")

    # NumPy interop: rank boxes by area (largest first) using the areas property.
    order = np.argsort(-oriented_boxes2d.areas)
    largest_first = datatypes.OrientedBoxes2D(oriented_boxes2d.data[order])
    logger.info(f"OrientedBoxes2D ranked by area (largest first): {largest_first.areas}")

    numpy_array = np.asarray(oriented_boxes2d)
    logger.info(f"NumPy array via __array__:\n{numpy_array}")

    # ======================= Visualize =========================================
    rr.init("oriented_boxes2d_example", spawn=True)
    datatypes.visualize(
        oriented_boxes2d,
        entity_path="/oriented_boxes2d/updated",
        label=["Updated Oriented Box2D 1", "Updated Oriented Box2D 2"],
    )
    datatypes.visualize(
        translated_oriented_boxes2d,
        entity_path="/oriented_boxes2d/translated",
        label=["Translated Oriented Box2D 1", "Translated Oriented Box2D 2"],
    )
    datatypes.visualize(
        rotated_oriented_boxes2d,
        entity_path="/oriented_boxes2d/rotated",
        label=["Rotated Oriented Box2D 1", "Rotated Oriented Box2D 2"],
    )

    # ======================= Serialize / Deserialize ===========================
    start = time.perf_counter()
    serialized = datatypes.serialize(oriented_boxes2d)
    serialization_ms = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    deserialized = datatypes.deserialize(serialized)["param_0"]
    deserialization_ms = (time.perf_counter() - start) * 1000

    logger.info(f"Deserialized OrientedBoxes2D: {deserialized}")
    logger.info(f"Round-trip successful: {oriented_boxes2d == deserialized}")
    logger.info(f"Serialization time: {serialization_ms:.3f} ms")
    logger.info(f"Deserialization time: {deserialization_ms:.3f} ms")


if __name__ == "__main__":
    oriented_boxes2d_example()
