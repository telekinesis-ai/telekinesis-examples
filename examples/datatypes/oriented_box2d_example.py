"""Demonstrates the Telekinesis OrientedBox2D datatype."""

import time

import numpy as np
import rerun as rr
from loguru import logger

from telekinesis import datatypes

def oriented_box2d_example():
    """Demonstrate creation, inspection, operations, visualization, and serialization."""

    # ======================= Create ============================================
    # OrientedBox2D format is CXCYWH = [cx, cy, width, height] + rotation [yaw_deg]
    coords = [0.5, 0.5, 0.5, 0.5, 30.0]
    oriented_box2d = datatypes.OrientedBox2D(coords)
    logger.info(f"Created OrientedBox2D: {oriented_box2d}")

    xyxy_coords = [1.0, 1.5, 3.5, 3.0, 45.0]
    oriented_box2d_from_xyxy = datatypes.OrientedBox2D.from_xyxy(xyxy_coords)
    logger.info(f"OrientedBox2D created from xyxy format: {oriented_box2d_from_xyxy}")

    xywh_coords = [1.0, 1.5, 2.5, 1.5, 45.0]
    oriented_box2d_from_xywh = datatypes.OrientedBox2D.from_xywh(xywh_coords)
    logger.info(f"OrientedBox2D created from xywh format: {oriented_box2d_from_xywh}")

    # ======================= Inspect ===========================================
    logger.info(f"data={oriented_box2d.data}")
    logger.info(f"dtype={oriented_box2d.dtype}")
    logger.info(f"ndim={oriented_box2d.ndim}")
    logger.info(f"shape={oriented_box2d.shape}")
    logger.info(f"size={oriented_box2d.size}")
    logger.info(f"center={oriented_box2d.center}")
    logger.info(f"dimensions={oriented_box2d.dimensions}")
    logger.info(f"area={oriented_box2d.area}")
    logger.info(f"rotation={oriented_box2d.rotation}")

    # ======================= Operations =========================================
    updated_coords = [2.0, 2.0, 1.5, 3.0, 45.0]
    oriented_box2d.data = updated_coords
    logger.info(f"Updated OrientedBox2D: {oriented_box2d}")

    xyxy_view = oriented_box2d.as_xyxy()
    logger.info(f"OrientedBox2D converted to xyxy format: {xyxy_view}")

    xywh_view = oriented_box2d.as_xywh()
    logger.info(f"OrientedBox2D converted to xywh format: {xywh_view}")

    oriented_box2d_copy = oriented_box2d.copy()
    logger.info(f"Copied OrientedBox2D: {oriented_box2d_copy}")

    # Returns the internal data as a NumPy array. If copy=True, returns a copy; otherwise, returns a view.
    oriented_box2d_numpy = oriented_box2d.to_numpy(copy=False)
    logger.info(f"NumPy OrientedBox2D:\n{oriented_box2d_numpy}")

    # Translate, scale, and rotate by operating on the underlying NumPy array directly.
    translation = [1.0, 1.0]
    translated_data = oriented_box2d.data.copy()
    translated_data[:2] += translation
    translated_oriented_box2d = datatypes.OrientedBox2D(translated_data)
    logger.info(f"Translated OrientedBox2D: {translated_oriented_box2d}")

    scale_factors = [1.5, 1.5]
    scaled_data = oriented_box2d.data.copy()
    scaled_data[2:4] *= np.asarray(scale_factors, dtype=np.float32)
    scaled_oriented_box2d = datatypes.OrientedBox2D(scaled_data)
    logger.info(f"Scaled OrientedBox2D: {scaled_oriented_box2d}")

    rotation_delta_deg = 15.0
    rotated_data = oriented_box2d.data.copy()
    rotated_data[4] += rotation_delta_deg
    rotated_oriented_box2d = datatypes.OrientedBox2D(rotated_data)
    logger.info(f"Rotated OrientedBox2D: {rotated_oriented_box2d}")

    numpy_array = np.asarray(oriented_box2d)
    logger.info(f"NumPy array via __array__: {numpy_array}")

    # ======================= Visualize =========================================
    rr.init("oriented_box2d_example", spawn=True)
    datatypes.visualize(oriented_box2d, entity_path="/oriented_box2d/updated", label="Updated Oriented Box2D")
    datatypes.visualize(
        translated_oriented_box2d,
        entity_path="/oriented_box2d/translated",
        label="Translated Oriented Box2D",
    )
    datatypes.visualize(
        scaled_oriented_box2d, entity_path="/oriented_box2d/scaled", label="Scaled Oriented Box2D"
    )
    datatypes.visualize(
        rotated_oriented_box2d, entity_path="/oriented_box2d/rotated", label="Rotated Oriented Box2D"
    )

    # ======================= Serialize / Deserialize ===========================
    start = time.perf_counter()
    serialized = datatypes.serialize(oriented_box2d)
    serialization_ms = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    deserialized = datatypes.deserialize(serialized)["param_0"]
    deserialization_ms = (time.perf_counter() - start) * 1000

    logger.info(f"Deserialized OrientedBox2D: {deserialized}")
    logger.info(f"Round-trip successful: {oriented_box2d == deserialized}")
    logger.info(f"Serialization time: {serialization_ms:.3f} ms")
    logger.info(f"Deserialization time: {deserialization_ms:.3f} ms")


if __name__ == "__main__":
    oriented_box2d_example()
