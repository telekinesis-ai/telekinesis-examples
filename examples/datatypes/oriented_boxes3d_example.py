"""Demonstrates the Telekinesis OrientedBoxes3D datatype."""

import itertools
import time

import numpy as np
import rerun as rr
from loguru import logger
from scipy.spatial.transform import Rotation

from telekinesis import datatypes

def oriented_boxes3d_example():
    """Demonstrate creation, access, visualization, update, format conversion, translate/scale/rotate transforms, NumPy corner computation, volume ranking, and serialization."""

    # ======================= Create ============================================
    # OrientedBoxes3D format is CXCYCZWHD = [[cx, cy, cz, width, height, depth], ...]
    # + rotation columns [roll_deg, pitch_deg, yaw_deg]
    box3d_1 = [0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 0.0, 0.0, 30.0]
    box3d_2 = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 45.0]
    boxes3d = datatypes.OrientedBoxes3D([box3d_1, box3d_2])

    logger.info(f"Original OrientedBoxes3D: {boxes3d}")

    # ======================= Inspect ===========================================
    logger.info(f"OrientedBoxes3D data: {boxes3d.data}")
    logger.info(
        f"dtype={boxes3d.dtype}, "
        f"ndim={boxes3d.ndim}, "
        f"shape={boxes3d.shape}, "
        f"dimensions={boxes3d.dimensions}, "
        f"volumes={boxes3d.volumes}, "
        f"centers={boxes3d.centers}, "
        f"rotations={boxes3d.rotations}"
    )
    # `.quaternions` was removed; derive them directly from `.rotations` when needed.
    quaternions = Rotation.from_euler("xyz", boxes3d.rotations, degrees=True).as_quat()
    logger.info(f"Derived quaternions [qx, qy, qz, qw] (via scipy): {quaternions}")

    # ======================= Visualize =========================================
    rr.init("oriented_box3d_example", spawn=True)
    datatypes.visualize(
        boxes3d,
        entity_path="/OrientedBox3D/oriented_boxes3d",
        label=["My Oriented Box3D 1", "My Oriented Box3D 2"],
    )

    # ======================= Update ============================================
    updated_data = boxes3d.data
    updated_data[0] = [2.0, 2.0, 2.0, 1.5, 1.0, 1.0, 0.0, 20.0, 0.0]
    boxes3d.data = updated_data
    logger.info(f"Updated OrientedBoxes3D: {boxes3d}")
    datatypes.visualize(
        boxes3d,
        entity_path="/OrientedBoxes3D/updated_oriented_box3d",
        label=["Updated Oriented Box3D 1", "Updated Oriented Box3D 2"],
    )

    # ======================= Alternate Construction =============================
    # Only the center/dimensions portion is reinterpreted; the trailing
    # rotation columns pass through unchanged.
    xyzxyz_coords = [
        [0.5, 1.0, 1.5, 3.5, 3.0, 2.5, 0.0, 20.0, 0.0],
        [1.0, 1.5, 2.0, 2.5, 2.5, 2.5, 0.0, 0.0, 45.0],
    ]
    boxes_from_xyzxyz = datatypes.OrientedBoxes3D.from_xyzxyz(xyzxyz_coords)
    logger.info(f"OrientedBoxes3D created from xyzxyz format: {boxes_from_xyzxyz}")

    xyzxyz_view = boxes3d.as_xyzxyz()
    logger.info(f"OrientedBoxes3D converted to xyzxyz format: {xyzxyz_view}")

    xyzwhd_coords = [
        [0.5, 1.0, 1.5, 3.0, 2.0, 1.0, 0.0, 20.0, 0.0],
        [1.0, 1.5, 2.0, 1.5, 1.0, 0.5, 0.0, 0.0, 45.0],
    ]
    boxes_from_xyzwhd = datatypes.OrientedBoxes3D.from_xyzwhd(xyzwhd_coords)
    logger.info(f"OrientedBoxes3D created from xyzwhd format: {boxes_from_xyzwhd}")

    xyzwhd_view = boxes3d.as_xyzwhd()
    logger.info(f"OrientedBoxes3D converted to xyzwhd format: {xyzwhd_view}")

    # ======================= NumPy Interop =====================================
    # Translate, scale, and rotate by operating on the underlying NumPy array directly.
    translated_data = boxes3d.data.copy()
    translated_data[:, :3] += [1.0, 1.0, 1.0]
    translated = datatypes.OrientedBoxes3D(translated_data)
    logger.info(f"Translated centers: {translated.centers} (was {boxes3d.centers})")
    datatypes.visualize(
        translated,
        entity_path="/OrientedBoxes3D/translated_oriented_box3d",
        label=["Translated Oriented Box3D 1", "Translated Oriented Box3D 2"],
    )

    scaled_data = boxes3d.data.copy()
    scaled_data[:, 3:6] *= 1.5
    scaled = datatypes.OrientedBoxes3D(scaled_data)
    logger.info(f"Scaled dimensions: {scaled.dimensions} (was {boxes3d.dimensions})")
    datatypes.visualize(
        scaled,
        entity_path="/OrientedBoxes3D/scaled_oriented_box3d",
        label=["Scaled Oriented Box3D 1", "Scaled Oriented Box3D 2"],
    )

    delta_rotation_deg = [15.0, 0.0, 0.0]
    current_rot = Rotation.from_euler("xyz", boxes3d.rotations, degrees=True)
    delta_rot = Rotation.from_euler("xyz", delta_rotation_deg, degrees=True)
    composed_deg = (delta_rot * current_rot).as_euler("xyz", degrees=True)

    rotated_data = boxes3d.data.copy()
    rotated_data[:, 6:9] = composed_deg
    rotated = datatypes.OrientedBoxes3D(rotated_data)
    logger.info(f"Rotated rotations: {rotated.rotations} (was {boxes3d.rotations})")
    datatypes.visualize(
        rotated,
        entity_path="/OrientedBoxes3D/rotated_oriented_box3d",
        label=["Rotated Oriented Box3D 1", "Rotated Oriented Box3D 2"],
    )

    # ======================= Rank by Volume ====================================
    order = np.argsort(-rotated.volumes)
    largest_first = datatypes.OrientedBoxes3D(rotated.data[order])
    logger.info(
        f"Boxes ranked by volume (largest first): {largest_first.volumes} (order: {order.tolist()})"
    )

    # ======================= Serialize / Deserialize ===========================
    start = time.perf_counter()
    serialized = datatypes.serialize(boxes3d)
    serialization_ms = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    deserialized = datatypes.deserialize(serialized)["param_0"]
    deserialization_ms = (time.perf_counter() - start) * 1000

    logger.info(f"Deserialized OrientedBoxes3D: {deserialized}")
    logger.info(f"Round-trip successful: {boxes3d == deserialized}")
    logger.info(f"Serialization time: {serialization_ms:.3f} ms")
    logger.info(f"Deserialization time: {deserialization_ms:.3f} ms")


if __name__ == "__main__":
    oriented_boxes3d_example()
