"""
Example script to demonstrate usage of OrientedBoxes2D datatype.
"""

import time

import numpy as np
from loguru import logger
import rerun as rr

from telekinesis import datatypes


def oriented_boxes2d_example():
    """
    Example function to demonstrate usage of OrientedBoxes2D datatype.
        - Create an OrientedBoxes2D data
        - Access the underlying box data
        - Visualize the OrientedBoxes2D data using Rerun
        - Update the underlying box data
        - Translate the box, returns a new OrientedBoxes2D object with updated center
        - Scale the box, returns a new OrientedBoxes2D object with updated size
        - Rotate the box, returns a new OrientedBoxes2D object with updated theta
        - Use with numpy to compute every box's rotated corner points, vectorized across the batch
        - Use with numpy to rank boxes by area via `np.argsort`
        - Serialize to PyArrow and back
    """
    # Create an OrientedBoxes2D data
    input_oriented_box2d_1 = [0.5, 0.5, 0.5, 0.5, 0.5]  # [cx, cy, w, h, theta]
    input_oriented_box2d_2 = [1.0, 1.0, 1.0, 1.0, 0.25]  # [cx, cy, w, h, theta]
    input_oriented_boxes2d = [input_oriented_box2d_1, input_oriented_box2d_2]
    my_oriented_boxes2d = datatypes.OrientedBoxes2D(input_oriented_boxes2d)
    logger.info(f"Original OrientedBoxes2D: {my_oriented_boxes2d}")

    # Access the underlying box data
    my_oriented_boxes2d_data = my_oriented_boxes2d.data
    my_oriented_boxes2d_shape = my_oriented_boxes2d.shape
    my_oriented_boxes2d_dtype = my_oriented_boxes2d.dtype
    my_oriented_boxes2d_ndim = my_oriented_boxes2d.ndim
    my_oriented_boxes2d_numpy = my_oriented_boxes2d.to_numpy()
    my_oriented_boxes2d_center = my_oriented_boxes2d.center
    my_oriented_boxes2d_area = my_oriented_boxes2d.area
    my_oriented_boxes2d_width = my_oriented_boxes2d.width
    my_oriented_boxes2d_height = my_oriented_boxes2d.height
    my_oriented_boxes2d_theta = my_oriented_boxes2d.theta

    logger.info("Visualizing with Rerun...")
    rr.init("oriented_box2d_example", spawn=True)
    datatypes.visualize(
        my_oriented_boxes2d,
        entity_path="/OrientedBox2D/my_oriented_boxes2d",
        label=["My Oriented Box2D 1", "My Oriented Box2D 2"],
    )

    logger.info(f"Underlying OrientedBoxes2D data: {my_oriented_boxes2d_data}")
    logger.info(f"Underlying OrientedBoxes2D data shape: {my_oriented_boxes2d_shape}")
    logger.info(f"Underlying OrientedBoxes2D data dtype: {my_oriented_boxes2d_dtype}")
    logger.info(f"Underlying OrientedBoxes2D data ndim: {my_oriented_boxes2d_ndim}")
    logger.info(f"Underlying OrientedBoxes2D data as numpy array: {my_oriented_boxes2d_numpy}")
    logger.info(f"Underlying OrientedBoxes2D center: {my_oriented_boxes2d_center}")
    logger.info(f"Underlying OrientedBoxes2D area: {my_oriented_boxes2d_area}")
    logger.info(f"Underlying OrientedBoxes2D width: {my_oriented_boxes2d_width}")
    logger.info(f"Underlying OrientedBoxes2D height: {my_oriented_boxes2d_height}")
    logger.info(f"Underlying OrientedBoxes2D theta: {my_oriented_boxes2d_theta}")

    # Update the my_oriented_box2d data
    my_oriented_boxes2d.data = [
        [2.0, 2.0, 1.5, 1.0, 1.0],
        [3.0, 3.0, 2.0, 1.5, 0.5],
    ]
    logger.info(f"Updated OrientedBoxes2D: {my_oriented_boxes2d}")
    datatypes.visualize(
        my_oriented_boxes2d,
        entity_path="/OrientedBoxes2D/my_updated_oriented_box2d",
        label=["Updated Oriented Box2D 1", "Updated Oriented Box2D 2"],
    )

    # Translate the box, returns a new OrientedBox2D object with updated center
    translated_oriented_box2d = my_oriented_boxes2d.translate([3.0, 3.0])
    logger.info(
        f"Translated OrientedBoxes2D center: {translated_oriented_box2d.center} "
        f"(was {my_oriented_boxes2d.center})"
    )
    datatypes.visualize(
        translated_oriented_box2d,
        entity_path="/OrientedBoxes2D/my_translated_oriented_box2d",
        label=["Translated Oriented Box2D 1", "Translated Oriented Box2D 2"],
    )

    # Scale the box, returns a new OrientedBox2D object with updated size
    # scaled_oriented_box2d = my_oriented_boxes2d.scale(1.5)
    # logger.info(
    #     f"Scaled OrientedBoxes2D width and height: {scaled_oriented_box2d.width} x {scaled_oriented_box2d.height} "
    #     f"(was {my_oriented_boxes2d.width} x {my_oriented_boxes2d.height})"
    # )
    # datatypes.visualize(
    #     scaled_oriented_box2d,
    #     entity_path="/OrientedBoxes2D/my_scaled_oriented_box2d",
    #     label=["Scaled Oriented Box2D 1", "Scaled Oriented Box2D 2"],
    # )

    # Rotate the box, returns a new OrientedBox2D object with theta increased by the delta
    rotated_oriented_box2d = my_oriented_boxes2d.rotate(0.25)
    logger.info(
        f"Rotated OrientedBoxes2D theta: {rotated_oriented_box2d.theta} "
        f"(was {my_oriented_boxes2d.theta})"
    )
    datatypes.visualize(
        rotated_oriented_box2d,
        entity_path="/OrientedBoxes2D/my_rotated_oriented_box2d",
        label=["Rotated Oriented Box2D 1", "Rotated Oriented Box2D 2"],
    )

    # Use with numpy to compute every box's rotated corner points at once --
    # vectorized across the batch (no Python loop over boxes), which is the
    # numpy pattern developers reach for by default when processing a batch.
    data = np.asarray(rotated_oriented_box2d)  # shape (N, 5)
    centers = data[:, :2]  # shape (N, 2)
    half_extents = data[:, 2:4] / 2  # shape (N, 2): [w/2, h/2] per box
    theta = data[:, 4]  # shape (N,)

    corner_signs = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=np.float32)  # (4, 2)
    local_corners = corner_signs[None, :, :] * half_extents[:, None, :]  # (N, 4, 2)

    cos_t, sin_t = np.cos(theta), np.sin(theta)
    rotation_matrices = np.stack(
        [np.stack([cos_t, -sin_t], axis=-1), np.stack([sin_t, cos_t], axis=-1)], axis=1
    )  # shape (N, 2, 2)

    # Batched matmul: `@` on 3D arrays applies one (4,2) @ (2,2) matmul per box.
    corners = local_corners @ rotation_matrices.transpose(0, 2, 1) + centers[:, None, :]
    logger.info(
        f"Rotated OrientedBoxes2D corners per box, world space, shape {corners.shape}:\n{corners}"
    )

    # Sanity-check the numpy corner math against the boxes' own `.area`
    edge_w = np.linalg.norm(corners[:, 1] - corners[:, 0], axis=-1)
    edge_h = np.linalg.norm(corners[:, 2] - corners[:, 1], axis=-1)
    logger.info(
        f"Area from numpy corners: {edge_w * edge_h} (matches .area: {rotated_oriented_box2d.area})"
    )

    # Get order of the boxes by area, largest first, using numpy's argsort.
    order = np.argsort(-rotated_oriented_box2d.area)  # indices, largest area first
    largest_first = datatypes.OrientedBoxes2D(rotated_oriented_box2d.data[order])
    logger.info(
        f"Boxes ranked by area (largest first): {largest_first.area} (order: {order.tolist()})"
    )

    # Serialize to PyArrow and back
    serialization_start_time = time.perf_counter()
    serialized = datatypes.serialize(my_oriented_boxes2d)
    serialization_end_time = time.perf_counter()

    deserialization_start_time = time.perf_counter()
    deserialized = datatypes.deserialize(serialized)["param_0"]
    deserialization_end_time = time.perf_counter()

    logger.info(
        f"Deserialized OrientedBoxes2D matches Original: {deserialized == my_oriented_boxes2d}"
    )
    logger.info(
        f"Serialization time: {(serialization_end_time - serialization_start_time) * 1000} ms"
    )
    logger.info(
        f"Deserialization time: {(deserialization_end_time - deserialization_start_time) * 1000} ms"
    )


if __name__ == "__main__":
    oriented_boxes2d_example()
