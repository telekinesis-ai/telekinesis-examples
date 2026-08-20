"""Demonstrates the Telekinesis OrientedBoxes3D datatype."""

import itertools
import time

import numpy as np
from scipy.spatial.transform import Rotation
from loguru import logger
import rerun as rr

from telekinesis import datatypes

def oriented_boxes3d_example():
    """Demonstrate creation, access, visualization, update, translate/scale/rotate transforms, NumPy corner computation, volume ranking, and serialization."""

    # ======================= Create ============================================
    box3d_1 = [0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 0.0, 0.0, 0.258819, 0.965926]
    box3d_2 = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.382683, 0.923880]
    boxes3d = datatypes.OrientedBoxes3D([box3d_1, box3d_2])

    logger.info(f"Original OrientedBoxes3D: {boxes3d}")

    # ======================= Inspect ===========================================
    logger.info(f"shape={boxes3d.shape}, dtype={boxes3d.dtype}, ndim={boxes3d.ndim}")
    logger.info(f"Underlying data: {boxes3d.data}")
    logger.info(f"NumPy array: {boxes3d.to_numpy()}")
    logger.info(
        f"center={boxes3d.center}, "
        f"volume={boxes3d.volume}, "
        f"width={boxes3d.width}, "
        f"height={boxes3d.height}, "
        f"depth={boxes3d.depth}"
    )
    logger.info(f"Quaternion: {boxes3d.data[:, 6:10]}")

    # ======================= Visualize =========================================
    rr.init("oriented_box3d_example", spawn=True)
    datatypes.visualize(
        boxes3d,
        entity_path="/OrientedBox3D/my_oriented_boxes3d",
        label=["My Oriented Box3D 1", "My Oriented Box3D 2"],
    )

    # ======================= Update ============================================
    updated_data = boxes3d.data
    updated_data[0] = [2.0, 2.0, 2.0, 1.5, 1.0, 1.0, 0.0, 0.173648, 0.0, 0.984808]
    boxes3d.data = updated_data
    logger.info(f"Updated OrientedBoxes3D: {boxes3d}")
    datatypes.visualize(
        boxes3d,
        entity_path="/OrientedBoxes3D/my_updated_oriented_box3d",
        label=["Updated Oriented Box3D 1", "Updated Oriented Box3D 2"],
    )

    # ======================= Translate =========================================
    translated = boxes3d.translate([1.0, 1.0, 1.0])
    logger.info(f"Translated center: {translated.center} (was {boxes3d.center})")
    datatypes.visualize(
        translated,
        entity_path="/OrientedBoxes3D/my_translated_oriented_box3d",
        label=["Translated Oriented Box3D 1", "Translated Oriented Box3D 2"],
    )

    # ======================= Scale =============================================
    scaled = boxes3d.scale(1.5)
    logger.info(
        f"Scaled width, height, depth: {scaled.width} x {scaled.height} x {scaled.depth} "
        f"(was {boxes3d.width} x {boxes3d.height} x {boxes3d.depth})"
    )
    datatypes.visualize(
        scaled,
        entity_path="/OrientedBoxes3D/my_scaled_oriented_box3d",
        label=["Scaled Oriented Box3D 1", "Scaled Oriented Box3D 2"],
    )

    # ======================= Rotate ============================================
    delta_quat = [0.130526, 0.0, 0.0, 0.991445]
    rotated = boxes3d.rotate(delta_quat)
    logger.info(
        f"Rotated quaternion: {rotated.data[:, 6:10]} (was {boxes3d.data[:, 6:10]})"
    )
    datatypes.visualize(
        rotated,
        entity_path="/OrientedBoxes3D/my_rotated_oriented_box3d",
        label=["Rotated Oriented Box3D 1", "Rotated Oriented Box3D 2"],
    )

    # ======================= NumPy Interop =====================================
    data = np.asarray(rotated)
    centers = data[:, :3]
    half_extents = data[:, 3:6] / 2
    quats_xyzw = data[:, 6:10]

    corner_signs = np.array(list(itertools.product([-1, 1], repeat=3)), dtype=np.float32)
    local_corners = corner_signs[None, :, :] * half_extents[:, None, :]

    rotation_matrices = Rotation.from_quat(quats_xyzw).as_matrix().astype(np.float32)

    corners = local_corners @ rotation_matrices.transpose(0, 2, 1) + centers[:, None, :]
    logger.info(f"Corners per box, world space, shape {corners.shape}:\n{corners}")

    edge_d = np.linalg.norm(corners[:, 1] - corners[:, 0], axis=-1)
    edge_h = np.linalg.norm(corners[:, 2] - corners[:, 0], axis=-1)
    edge_w = np.linalg.norm(corners[:, 4] - corners[:, 0], axis=-1)
    logger.info(
        f"Volume from numpy corners: {edge_w * edge_h * edge_d} (matches .volume: {rotated.volume})"
    )

    # ======================= Rank by Volume ====================================
    order = np.argsort(-rotated.volume)
    largest_first = datatypes.OrientedBoxes3D(rotated.data[order])
    logger.info(
        f"Boxes ranked by volume (largest first): {largest_first.volume} (order: {order.tolist()})"
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
