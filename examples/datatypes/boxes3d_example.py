"""Demonstrates the Telekinesis Boxes3D datatype."""

import time

import numpy as np
import rerun as rr
from loguru import logger

from telekinesis import datatypes

def boxes3d_example():
    """Demonstrate creation, inspection, operations, visualization, and serialization."""

    # ======================= Create ============================================
    # Boxes3D format is CXCYCZWHD = [[cx, cy, cz, width, height, depth], ...]
    box3d_1 = [[0, 0, 0, 1, 1, 1]]
    box3d_2 = [[2, 2, 2, 3, 3, 3]]
    coords = np.concatenate([box3d_1, box3d_2], axis=0)
    boxes3d = datatypes.Boxes3D(coords)
    logger.info(f"Original Boxes3D: {boxes3d}")

    xyzxyz_coords = [[0, 0, 0, 1, 1, 1], [2, 2, 2, 3, 3, 3]]
    boxes3d_from_xyzxyz = datatypes.Boxes3D.from_xyzxyz(xyzxyz_coords)
    logger.info(f"Boxes3D created from xyzxyz format: {boxes3d_from_xyzxyz}")

    xyzwhd_coords = [[0, 0, 0, 1, 1, 1], [2, 2, 2, 1, 1, 1]]
    boxes3d_from_xyzwhd = datatypes.Boxes3D.from_xyzwhd(xyzwhd_coords)
    logger.info(f"Boxes3D created from xyzwhd format: {boxes3d_from_xyzwhd}")

    # ======================= Inspect ===========================================
    logger.info(f"data={boxes3d.data}")
    logger.info(f"dtype={boxes3d.dtype}")
    logger.info(f"ndim={boxes3d.ndim}")
    logger.info(f"shape={boxes3d.shape}")
    logger.info(f"size={boxes3d.size}")
    logger.info(f"dimensions={boxes3d.dimensions}")
    logger.info(f"volumes={boxes3d.volumes}")
    logger.info(f"centers={boxes3d.centers}")

    # ======================= Operations =========================================
    updated_box = [3, 3, 3, 1, 1, 1]
    data = boxes3d.data
    data[1] = updated_box
    boxes3d.data = data
    logger.info(f"Updated Boxes3D: {boxes3d}")

    xyzxyz_view = boxes3d.as_xyzxyz()
    logger.info(f"Boxes3D converted to xyzxyz format: {xyzxyz_view}")

    xyzwhd_view = boxes3d.as_xyzwhd()
    logger.info(f"Boxes3D converted to xyzwhd format: {xyzwhd_view}")

    boxes3d_copy = boxes3d.copy()
    logger.info(f"Copied Boxes3D: {boxes3d_copy}")

    # Returns the internal data as a NumPy array. If copy=True, returns a copy; otherwise, returns a view.
    boxes3d_numpy = boxes3d.to_numpy(copy=False)
    logger.info(f"NumPy Boxes3D:\n{boxes3d_numpy}")

    numpy_boxes3d = np.asarray(boxes3d)
    logger.info(f"Boxes3D via __array__:\n{numpy_boxes3d}")

    logger.info(f"Number of boxes: {len(boxes3d)}")
    first_box3d = boxes3d[0]
    sub_batch = boxes3d[1:]
    logger.info(f"First box: {first_box3d}")
    logger.info(f"Sub-batch [1:]: {sub_batch}")

    # Translate and scale by operating on the underlying NumPy array directly.
    translation = [2, 3, 1]
    translated_data = boxes3d.data.copy()
    translated_data[:, :3] += translation
    translated_boxes3d = datatypes.Boxes3D(translated_data)
    logger.info(f"Translated Boxes3D: {translated_boxes3d}")

    scale_factors = [0.5, 0.5, 0.5]
    scaled_data = boxes3d.data.copy()
    scaled_data[:, 3:] *= np.asarray(scale_factors, dtype=np.float32)
    scaled_boxes3d = datatypes.Boxes3D(scaled_data)
    logger.info(f"Scaled Boxes3D: {scaled_boxes3d}")

    # ======================= Visualize =========================================
    rr.init("boxes3d_example", spawn=True)
    datatypes.visualize(
        boxes3d, entity_path="/boxes3d/updated", label=["Updated Box3D 1", "Updated Box3D 2"]
    )
    datatypes.visualize(
        translated_boxes3d,
        entity_path="/boxes3d/translated",
        label=["Translated Box3D 1", "Translated Box3D 2"],
    )
    datatypes.visualize(
        scaled_boxes3d,
        entity_path="/boxes3d/scaled",
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
