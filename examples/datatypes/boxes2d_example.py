"""Demonstrates the Telekinesis Boxes2D datatype."""

import time

import numpy as np
import rerun as rr
from loguru import logger

from telekinesis import datatypes

def boxes2d_example():
    """Demonstrate creation, inspection, operations, visualization, and serialization."""

    # ======================= Create ============================================
    # Boxes2D format is CXCYWH = [[cx, cy, width, height], ...]
    box2d_1 = [[1, 2.5, 3, 3]]
    box2d_2 = [[4, 5, 2, 1]]
    coords = np.concatenate([box2d_1, box2d_2], axis=0)
    boxes2d = datatypes.Boxes2D(coords)
    logger.info(f"Original Boxes2D: {boxes2d}")

    xyxy_coords = [[1.0, 1.5, 4.0, 4.5], [3.0, 4.5, 5.0, 5.5]]
    boxes2d_from_xyxy = datatypes.Boxes2D.from_xyxy(xyxy_coords)
    logger.info(f"Boxes2D created from xyxy format: {boxes2d_from_xyxy}")

    xywh_coords = [[1.0, 1.5, 3.0, 3.0], [3.0, 4.5, 2.0, 1.0]]
    boxes2d_from_xywh = datatypes.Boxes2D.from_xywh(xywh_coords)
    logger.info(f"Boxes2D created from xywh format: {boxes2d_from_xywh}")

    # ======================= Inspect ===========================================
    logger.info(f"data={boxes2d.data}")
    logger.info(f"dtype={boxes2d.dtype}")
    logger.info(f"ndim={boxes2d.ndim}")
    logger.info(f"shape={boxes2d.shape}")
    logger.info(f"size={boxes2d.size}")
    logger.info(f"dimensions={boxes2d.dimensions}")
    logger.info(f"areas={boxes2d.areas}")
    logger.info(f"centers={boxes2d.centers}")

    # ======================= Operations =========================================
    updated_box = [3, 4, 3, 5]
    data = boxes2d.data
    data[1] = updated_box
    boxes2d.data = data
    logger.info(f"Updated Boxes2D: {boxes2d}")

    xyxy_view = boxes2d.as_xyxy()
    logger.info(f"Boxes2D converted to xyxy format: {xyxy_view}")

    xywh_view = boxes2d.as_xywh()
    logger.info(f"Boxes2D converted to xywh format: {xywh_view}")

    boxes2d_copy = boxes2d.copy()
    logger.info(f"Copied Boxes2D: {boxes2d_copy}")

    # Returns the internal data as a NumPy array. If copy=True, returns a copy; otherwise, returns a view.
    boxes2d_numpy = boxes2d.to_numpy(copy=False)
    logger.info(f"NumPy Boxes2D:\n{boxes2d_numpy}")

    numpy_boxes2d = np.asarray(boxes2d)
    logger.info(f"Boxes2D via __array__:\n{numpy_boxes2d}")

    logger.info(f"Number of boxes: {len(boxes2d)}")
    first_box2d = boxes2d[0]
    sub_batch = boxes2d[1:]
    logger.info(f"First box: {first_box2d}")
    logger.info(f"Sub-batch [1:]: {sub_batch}")

    # Translate and scale by operating on the underlying NumPy array directly.
    translation = [2, 3]
    translated_data = boxes2d.data.copy()
    translated_data[:, :2] += translation
    translated_boxes2d = datatypes.Boxes2D(translated_data)
    logger.info(f"Translated Boxes2D: {translated_boxes2d}")

    scale_factors = [0.5, 0.5]
    scaled_data = boxes2d.data.copy()
    scaled_data[:, 2:] *= np.asarray(scale_factors, dtype=np.float32)
    scaled_boxes2d = datatypes.Boxes2D(scaled_data)
    logger.info(f"Scaled Boxes2D: {scaled_boxes2d}")

    # ======================= Visualize =========================================
    rr.init("boxes2d_example", spawn=True)
    datatypes.visualize(
        boxes2d, entity_path="/boxes2d/updated", label=["Updated Box2D 1", "Updated Box2D 2"]
    )
    datatypes.visualize(
        translated_boxes2d,
        entity_path="/boxes2d/translated",
        label=["Translated Box2D 1", "Translated Box2D 2"],
    )
    datatypes.visualize(
        scaled_boxes2d,
        entity_path="/boxes2d/scaled",
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
