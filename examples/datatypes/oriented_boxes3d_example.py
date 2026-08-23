"""Demonstrates the Telekinesis OrientedBoxes3D datatype."""

import time

import numpy as np
import rerun as rr
from loguru import logger

from telekinesis import datatypes

def oriented_boxes3d_example():
    """Demonstrate creation, inspection, operations, visualization, and serialization."""

    # ======================= Create ============================================
    # OrientedBoxes3D format is CXCYCZWHD = [[cx, cy, cz, width, height, depth], ...]
    # + rotation columns [roll_deg, pitch_deg, yaw_deg] (Euler XYZ, in degrees)
    oriented_box3d_1 = [0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 0.0, 0.0, 30.0]
    oriented_box3d_2 = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 45.0]
    oriented_boxes3d = datatypes.OrientedBoxes3D([oriented_box3d_1, oriented_box3d_2])
    logger.info(f"Created OrientedBoxes3D: {oriented_boxes3d}")

    xyzxyz_coords = [
        [0.5, 1.0, 1.5, 3.5, 3.0, 2.5, 0.0, 20.0, 0.0],
        [1.0, 1.5, 2.0, 2.5, 2.5, 2.5, 0.0, 0.0, 45.0],
    ]
    oriented_boxes3d_from_xyzxyz = datatypes.OrientedBoxes3D.from_xyzxyz(xyzxyz_coords)
    logger.info(f"OrientedBoxes3D created from xyzxyz format: {oriented_boxes3d_from_xyzxyz}")

    xyzwhd_coords = [
        [0.5, 1.0, 1.5, 3.0, 2.0, 1.0, 0.0, 20.0, 0.0],
        [1.0, 1.5, 2.0, 1.5, 1.0, 0.5, 0.0, 0.0, 45.0],
    ]
    oriented_boxes3d_from_xyzwhd = datatypes.OrientedBoxes3D.from_xyzwhd(xyzwhd_coords)
    logger.info(f"OrientedBoxes3D created from xyzwhd format: {oriented_boxes3d_from_xyzwhd}")

    # ======================= Inspect ===========================================
    logger.info(f"data={oriented_boxes3d.data}")
    logger.info(f"dtype={oriented_boxes3d.dtype}")
    logger.info(f"ndim={oriented_boxes3d.ndim}")
    logger.info(f"shape={oriented_boxes3d.shape}")
    logger.info(f"size={oriented_boxes3d.size}")
    logger.info(f"length={len(oriented_boxes3d)}")
    logger.info(f"centers={oriented_boxes3d.centers}")
    logger.info(f"dimensions={oriented_boxes3d.dimensions}")
    logger.info(f"volumes={oriented_boxes3d.volumes}")
    logger.info(f"rotations={oriented_boxes3d.rotations}")

    # ======================= Operations =========================================
    updated_data = [
        [2.0, 2.0, 2.0, 1.5, 1.0, 1.0, 0.0, 20.0, 0.0],
        [3.0, 3.0, 3.0, 1.0, 1.0, 1.0, 0.0, 0.0, 45.0],
    ]
    oriented_boxes3d.data = updated_data
    logger.info(f"Updated OrientedBoxes3D: {oriented_boxes3d}")

    xyzxyz_view = oriented_boxes3d.as_xyzxyz()
    logger.info(f"OrientedBoxes3D converted to xyzxyz format: {xyzxyz_view}")

    xyzwhd_view = oriented_boxes3d.as_xyzwhd()
    logger.info(f"OrientedBoxes3D converted to xyzwhd format: {xyzwhd_view}")

    first_oriented_box3d = oriented_boxes3d[0]
    logger.info(f"First OrientedBox3D (index 0): {first_oriented_box3d}")

    sub_batch = oriented_boxes3d[1:]
    logger.info(f"Sub-batch of OrientedBoxes3D [1:]: {sub_batch}")

    oriented_boxes3d_copy = oriented_boxes3d.copy()
    logger.info(f"Copied OrientedBoxes3D: {oriented_boxes3d_copy}")

    # Returns the internal data as a NumPy array. If copy=True, returns a copy; otherwise, returns a view.
    oriented_boxes3d_numpy = oriented_boxes3d.to_numpy(copy=False)
    logger.info(f"NumPy OrientedBoxes3D:\n{oriented_boxes3d_numpy}")

    # Translate, scale, and rotate by operating on the underlying NumPy array directly.
    translation = [1.0, 1.0, 1.0]
    translated_data = oriented_boxes3d.data.copy()
    translated_data[:, :3] += translation
    translated_oriented_boxes3d = datatypes.OrientedBoxes3D(translated_data)
    logger.info(f"Translated OrientedBoxes3D: {translated_oriented_boxes3d}")

    scale_factors = [1.5, 1.5, 1.5]
    scaled_data = oriented_boxes3d.data.copy()
    scaled_data[:, 3:6] *= np.asarray(scale_factors, dtype=np.float32)
    scaled_oriented_boxes3d = datatypes.OrientedBoxes3D(scaled_data)
    logger.info(f"Scaled OrientedBoxes3D: {scaled_oriented_boxes3d}")

    # `rotations` stores Euler-XYZ degrees natively, so a rotation delta is
    # applied by adding directly to the [roll_deg, pitch_deg, yaw_deg] columns.
    rotation_delta_deg = [0.0, 0.0, 15.0]
    rotated_data = oriented_boxes3d.data.copy()
    rotated_data[:, 6:9] += np.asarray(rotation_delta_deg, dtype=np.float32)
    rotated_oriented_boxes3d = datatypes.OrientedBoxes3D(rotated_data)
    logger.info(f"Rotated OrientedBoxes3D: {rotated_oriented_boxes3d}")

    # NumPy interop: rank boxes by volume (largest first) using the volumes property.
    order = np.argsort(-oriented_boxes3d.volumes)
    largest_first = datatypes.OrientedBoxes3D(oriented_boxes3d.data[order])
    logger.info(f"OrientedBoxes3D ranked by volume (largest first): {largest_first.volumes}")

    numpy_array = np.asarray(oriented_boxes3d)
    logger.info(f"NumPy array via __array__:\n{numpy_array}")

    # ======================= Visualize =========================================
    rr.init("oriented_boxes3d_example", spawn=True)
    datatypes.visualize(
        oriented_boxes3d,
        entity_path="/oriented_boxes3d/updated",
        label=["Updated Oriented Box3D 1", "Updated Oriented Box3D 2"],
    )
    datatypes.visualize(
        translated_oriented_boxes3d,
        entity_path="/oriented_boxes3d/translated",
        label=["Translated Oriented Box3D 1", "Translated Oriented Box3D 2"],
    )
    datatypes.visualize(
        scaled_oriented_boxes3d,
        entity_path="/oriented_boxes3d/scaled",
        label=["Scaled Oriented Box3D 1", "Scaled Oriented Box3D 2"],
    )
    datatypes.visualize(
        rotated_oriented_boxes3d,
        entity_path="/oriented_boxes3d/rotated",
        label=["Rotated Oriented Box3D 1", "Rotated Oriented Box3D 2"],
    )

    # ======================= Serialize / Deserialize ===========================
    start = time.perf_counter()
    serialized = datatypes.serialize(oriented_boxes3d)
    serialization_ms = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    deserialized = datatypes.deserialize(serialized)["param_0"]
    deserialization_ms = (time.perf_counter() - start) * 1000

    logger.info(f"Deserialized OrientedBoxes3D: {deserialized}")
    logger.info(f"Round-trip successful: {oriented_boxes3d == deserialized}")
    logger.info(f"Serialization time: {serialization_ms:.3f} ms")
    logger.info(f"Deserialization time: {deserialization_ms:.3f} ms")


if __name__ == "__main__":
    oriented_boxes3d_example()
