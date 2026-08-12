"""
Example script to demonstrate usage of OrientedBoxes3D datatype.
"""

import itertools
import time

import numpy as np
from scipy.spatial.transform import Rotation
from loguru import logger
import rerun as rr

from telekinesis import datatypes


def oriented_boxes3d_example():
    """
    Example function to demonstrate usage of OrientedBoxes3D datatype.
        - Create an OrientedBoxes3D data
        - Access the underlying box data
        - Visualize the OrientedBoxes3D data using Rerun
        - Update the underlying box data
        - Translate the box, returns a new OrientedBoxes3D object with updated center
        - Scale the box, returns a new OrientedBoxes3D object with updated size
        - Rotate the box, returns a new OrientedBoxes3D object with a composed quaternion
        - Use with numpy to compute every box's rotated corner points, vectorized across the batch
        - Use with numpy to rank boxes by volume via `np.argsort`
        - Serialize to PyArrow and back
    """
    # Create an OrientedBoxes3D data. Quaternions are [qx, qy, qz, qw] (scalar-last),
    # so a rotation by angle `a` about a unit axis is [axis * sin(a/2), cos(a/2)].
    # Box 1: 30 deg yaw about Z. Box 2: 45 deg yaw about Z.
    input_oriented_box3d_1 = [0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 0.0, 0.0, 0.258819, 0.965926]
    input_oriented_box3d_2 = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.382683, 0.923880]
    input_oriented_boxes3d = [input_oriented_box3d_1, input_oriented_box3d_2]
    my_oriented_boxes3d = datatypes.OrientedBoxes3D(input_oriented_boxes3d)
    logger.info(f"Original OrientedBoxes3D: {my_oriented_boxes3d}")

    # Access the underlying box data
    my_oriented_boxes3d_data = my_oriented_boxes3d.data
    my_oriented_boxes3d_shape = my_oriented_boxes3d.shape
    my_oriented_boxes3d_dtype = my_oriented_boxes3d.dtype
    my_oriented_boxes3d_ndim = my_oriented_boxes3d.ndim
    my_oriented_boxes3d_numpy = my_oriented_boxes3d.to_numpy()
    my_oriented_boxes3d_center = my_oriented_boxes3d.center
    my_oriented_boxes3d_volume = my_oriented_boxes3d.volume
    my_oriented_boxes3d_width = my_oriented_boxes3d.width
    my_oriented_boxes3d_height = my_oriented_boxes3d.height
    my_oriented_boxes3d_depth = my_oriented_boxes3d.depth
    # No dedicated `.quaternion` property -- read it via `data[:, 6:10]` (`[qx, qy, qz, qw]`).
    my_oriented_boxes3d_quaternion = my_oriented_boxes3d.data[:, 6:10]

    logger.info("Visualizing with Rerun...")
    rr.init("oriented_box3d_example", spawn=True)
    datatypes.visualize(
        my_oriented_boxes3d,
        entity_path="/OrientedBox3D/my_oriented_boxes3d",
        label=["My Oriented Box3D 1", "My Oriented Box3D 2"],
    )

    logger.info(f"Underlying OrientedBoxes3D data: {my_oriented_boxes3d_data}")
    logger.info(f"Underlying OrientedBoxes3D data shape: {my_oriented_boxes3d_shape}")
    logger.info(f"Underlying OrientedBoxes3D data dtype: {my_oriented_boxes3d_dtype}")
    logger.info(f"Underlying OrientedBoxes3D data ndim: {my_oriented_boxes3d_ndim}")
    logger.info(f"Underlying OrientedBoxes3D data as numpy array: {my_oriented_boxes3d_numpy}")
    logger.info(f"Underlying OrientedBoxes3D center: {my_oriented_boxes3d_center}")
    logger.info(f"Underlying OrientedBoxes3D volume: {my_oriented_boxes3d_volume}")
    logger.info(f"Underlying OrientedBoxes3D width: {my_oriented_boxes3d_width}")
    logger.info(f"Underlying OrientedBoxes3D height: {my_oriented_boxes3d_height}")
    logger.info(f"Underlying OrientedBoxes3D depth: {my_oriented_boxes3d_depth}")
    logger.info(f"Underlying OrientedBoxes3D quaternion: {my_oriented_boxes3d_quaternion}")

    # Update the my_oriented_box3d data. New rotation for box 1: 20 deg pitch about Y.
    my_updated_oriented_boxes3d = my_oriented_boxes3d.data
    my_updated_oriented_boxes3d[0] = [2.0, 2.0, 2.0, 1.5, 1.0, 1.0, 0.0, 0.173648, 0.0, 0.984808]
    my_oriented_boxes3d.data = my_updated_oriented_boxes3d
    logger.info(f"Updated OrientedBoxes3D: {my_oriented_boxes3d}")
    datatypes.visualize(
        my_oriented_boxes3d,
        entity_path="/OrientedBoxes3D/my_updated_oriented_box3d",
        label=["Updated Oriented Box3D 1", "Updated Oriented Box3D 2"],
    )

    # Translate the box, returns a new OrientedBoxes3D object with updated center
    translated_oriented_box3d = my_oriented_boxes3d.translate([1.0, 1.0, 1.0])
    logger.info(
        f"Translated OrientedBoxes3D center: {translated_oriented_box3d.center} "
        f"(was {my_oriented_boxes3d.center})"
    )
    datatypes.visualize(
        translated_oriented_box3d,
        entity_path="/OrientedBoxes3D/my_translated_oriented_box3d",
        label=["Translated Oriented Box3D 1", "Translated Oriented Box3D 2"],
    )

    # Scale the box, returns a new OrientedBoxes3D object with updated size
    scaled_oriented_box3d = my_oriented_boxes3d.scale(1.5)
    logger.info(
        f"Scaled OrientedBoxes3D width, height, and depth: {scaled_oriented_box3d.width} x {scaled_oriented_box3d.height} x {scaled_oriented_box3d.depth} "
        f"(was {my_oriented_boxes3d.width} x {my_oriented_boxes3d.height} x {my_oriented_boxes3d.depth})"
    )
    datatypes.visualize(
        scaled_oriented_box3d,
        entity_path="/OrientedBoxes3D/my_scaled_oriented_box3d",
        label=["Scaled Oriented Box3D 1", "Scaled Oriented Box3D 2"],
    )

    # Rotate the box, returns a new OrientedBoxes3D object with an additional
    # rotation quaternion composed on top of each box's existing orientation.
    # Delta here is ~15 deg roll about X.
    rotation_delta_quat = [0.130526, 0.0, 0.0, 0.991445]
    rotated_oriented_box3d = my_oriented_boxes3d.rotate(rotation_delta_quat)
    logger.info(
        f"Rotated OrientedBoxes3D quaternion: {rotated_oriented_box3d.data[:, 6:10]} "
        f"(was {my_oriented_boxes3d.data[:, 6:10]})"
    )
    datatypes.visualize(
        rotated_oriented_box3d,
        entity_path="/OrientedBoxes3D/my_rotated_oriented_box3d",
        label=["Rotated Oriented Box3D 1", "Rotated Oriented Box3D 2"],
    )

    # Use with numpy to compute every box's rotated corner points at once --
    # vectorized across the batch (no Python loop over boxes), which is the
    # numpy pattern developers reach for by default when processing a batch.
    data = np.asarray(rotated_oriented_box3d)  # shape (N, 10)
    centers = data[:, :3]  # shape (N, 3)
    half_extents = data[:, 3:6] / 2  # shape (N, 3): [w/2, h/2, d/2] per box
    quats_xyzw = data[:, 6:10]  # shape (N, 4): [qx, qy, qz, qw]

    # 8 local corners of a box: every +/- combination of its half-extents.
    corner_signs = np.array(list(itertools.product([-1, 1], repeat=3)), dtype=np.float32)  # (8, 3)
    local_corners = corner_signs[None, :, :] * half_extents[:, None, :]  # (N, 8, 3)

    rotation_matrices = Rotation.from_quat(quats_xyzw).as_matrix().astype(np.float32)  # (N, 3, 3)

    # Batched matmul: `@` on 3D arrays applies one (8,3) @ (3,3) matmul per box.
    corners = local_corners @ rotation_matrices.transpose(0, 2, 1) + centers[:, None, :]
    logger.info(
        f"Rotated OrientedBoxes3D corners per box, world space, shape {corners.shape}:\n{corners}"
    )

    # Sanity-check the numpy corner math against the boxes' own `.volume`. With
    # corner_signs ordered by itertools.product, corners 0/1 differ only in z,
    # 0/2 only in y, and 0/4 only in x -- so their world-space distances recover
    # depth/height/width regardless of rotation (rotation preserves length).
    edge_d = np.linalg.norm(corners[:, 1] - corners[:, 0], axis=-1)
    edge_h = np.linalg.norm(corners[:, 2] - corners[:, 0], axis=-1)
    edge_w = np.linalg.norm(corners[:, 4] - corners[:, 0], axis=-1)
    logger.info(
        f"Volume from numpy corners: {edge_w * edge_h * edge_d} (matches .volume: {rotated_oriented_box3d.volume})"
    )

    # Another widely-used numpy pattern: rank boxes with `np.argsort` -- the
    # same building block behind confidence-ranking and non-max suppression.
    order = np.argsort(-rotated_oriented_box3d.volume)  # indices, largest volume first
    largest_first = datatypes.OrientedBoxes3D(rotated_oriented_box3d.data[order])
    logger.info(
        f"Boxes ranked by volume (largest first): {largest_first.volume} (order: {order.tolist()})"
    )

    # Serialize to PyArrow and back
    serialization_start_time = time.perf_counter()
    serialized = datatypes.serialize(my_oriented_boxes3d)
    serialization_end_time = time.perf_counter()

    deserialization_start_time = time.perf_counter()
    deserialized = datatypes.deserialize(serialized)["param_0"]
    deserialization_end_time = time.perf_counter()

    logger.info(
        f"Deserialized OrientedBoxes3D matches Original: {deserialized == my_oriented_boxes3d}"
    )
    logger.info(
        f"Serialization time: {(serialization_end_time - serialization_start_time) * 1000} ms"
    )
    logger.info(
        f"Deserialization time: {(deserialization_end_time - deserialization_start_time) * 1000} ms"
    )


if __name__ == "__main__":
    oriented_boxes3d_example()
