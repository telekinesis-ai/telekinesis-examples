"""
Example script to demonstrate usage of OrientedBox2D datatype.
"""

import time

import numpy as np
from loguru import logger
import rerun as rr

from telekinesis import datatypes


def oriented_box2d_example():
    """
    Example function to demonstrate usage of OrientedBox2D datatype.
        - Create an OrientedBox2D data
        - Access the underlying box data
        - Visualize the OrientedBox2D data using Rerun
        - Update the underlying box data
        - Translate the box, returns a new OrientedBox2D object with updated center
        - Scale the box, returns a new OrientedBox2D object with updated size
        - Rotate the box, returns a new OrientedBox2D object with updated theta
        - Use with numpy to compute the box's rotated corner points
        - Serialize to PyArrow and back
    """
    # Create an OrientedBox2D data
    my_oriented_box2d = datatypes.OrientedBox2D([0.5, 0.5, 0.5, 0.5, 0.5])
    logger.info(f"Original OrientedBox2D: {my_oriented_box2d}")

    # Access the underlying box data
    my_oriented_box2d_data = my_oriented_box2d.data
    my_oriented_box2d_shape = my_oriented_box2d.shape
    my_oriented_box2d_dtype = my_oriented_box2d.dtype
    my_oriented_box2d_ndim = my_oriented_box2d.ndim
    my_oriented_box2d_numpy = my_oriented_box2d.to_numpy()
    my_oriented_box2d_center = my_oriented_box2d.center
    my_oriented_box2d_area = my_oriented_box2d.area
    my_oriented_box2d_width = my_oriented_box2d.width
    my_oriented_box2d_height = my_oriented_box2d.height
    my_oriented_box2d_theta = my_oriented_box2d.theta

    logger.info("Visualizing with Rerun...")
    rr.init("oriented_box2d_example", spawn=True)
    datatypes.visualize(
        my_oriented_box2d,
        entity_path="/OrientedBox2D/my_oriented_box2d",
        label="My Oriented Box2D",
    )

    logger.info(f"Underlying OrientedBox2D data: {my_oriented_box2d_data}")
    logger.info(f"Underlying OrientedBox2D data shape: {my_oriented_box2d_shape}")
    logger.info(f"Underlying OrientedBox2D data dtype: {my_oriented_box2d_dtype}")
    logger.info(f"Underlying OrientedBox2D data ndim: {my_oriented_box2d_ndim}")
    logger.info(f"Underlying OrientedBox2D data as numpy array: {my_oriented_box2d_numpy}")
    logger.info(f"Underlying OrientedBox2D center: {my_oriented_box2d_center}")
    logger.info(f"Underlying OrientedBox2D area: {my_oriented_box2d_area}")
    logger.info(f"Underlying OrientedBox2D width: {my_oriented_box2d_width}")
    logger.info(f"Underlying OrientedBox2D height: {my_oriented_box2d_height}")
    logger.info(f"Underlying OrientedBox2D theta: {my_oriented_box2d_theta}")

    # Update the my_oriented_box2d data
    my_oriented_box2d.data = [2.0, 2.0, 1.5, 1.0, 1.0]
    logger.info(f"Updated OrientedBox2D: {my_oriented_box2d}")
    datatypes.visualize(
        my_oriented_box2d,
        entity_path="/OrientedBox2D/my_updated_oriented_box2d",
        label="Updated Oriented Box2D",
    )

    # Translate the box, returns a new OrientedBox2D object with updated center
    translated_oriented_box2d = my_oriented_box2d.translate([1.0, 1.0])
    logger.info(
        f"Translated OrientedBox2D center: {translated_oriented_box2d.center} "
        f"(was {my_oriented_box2d.center})"
    )
    datatypes.visualize(
        translated_oriented_box2d,
        entity_path="/OrientedBox2D/my_translated_oriented_box2d",
        label="Translated Oriented Box2D",
    )

    # Scale the box, returns a new OrientedBox2D object with updated size
    scaled_oriented_box2d = my_oriented_box2d.scale(1.5)
    logger.info(
        f"Scaled OrientedBox2D width and height: {scaled_oriented_box2d.width} x {scaled_oriented_box2d.height} "
        f"(was {my_oriented_box2d.width} x {my_oriented_box2d.height})"
    )
    datatypes.visualize(
        scaled_oriented_box2d,
        entity_path="/OrientedBox2D/my_scaled_oriented_box2d",
        label="Scaled Oriented Box2D",
    )

    # Rotate the box, returns a new OrientedBox2D object with theta increased by the delta
    rotated_oriented_box2d = my_oriented_box2d.rotate(0.25)
    logger.info(
        f"Rotated OrientedBox2D theta: {rotated_oriented_box2d.theta} "
        f"(was {my_oriented_box2d.theta})"
    )
    datatypes.visualize(
        rotated_oriented_box2d,
        entity_path="/OrientedBox2D/my_rotated_oriented_box2d",
        label="Rotated Oriented Box2D",
    )

    # Use with numpy to compute the box's rotated corner points.
    cx, cy, w, h, theta = np.asarray(rotated_oriented_box2d)
    local_corners = np.array(
        [[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]], dtype=np.float32
    )
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    rotation_matrix = np.array([[cos_t, -sin_t], [sin_t, cos_t]], dtype=np.float32)
    corners = local_corners @ rotation_matrix.T + np.array([cx, cy], dtype=np.float32)
    logger.info(f"Rotated OrientedBox2D corners (world space, [x, y] per row):\n{corners}")

    # Sanity-check the numpy corner math against the box's own `.area`
    edge_w = np.linalg.norm(corners[1] - corners[0])
    edge_h = np.linalg.norm(corners[2] - corners[1])
    logger.info(
        f"Area from numpy corners: {edge_w * edge_h} (matches .area: {rotated_oriented_box2d.area})"
    )

    # Serialize to PyArrow and back
    serialization_start_time = time.perf_counter()
    serialized = datatypes.serialize(my_oriented_box2d)
    serialization_end_time = time.perf_counter()

    deserialization_start_time = time.perf_counter()
    deserialized = datatypes.deserialize(serialized)["param_0"]
    deserialization_end_time = time.perf_counter()

    logger.info(f"Deserialized OrientedBox2D matches Original: {deserialized == my_oriented_box2d}")
    logger.info(
        f"Serialization time: {(serialization_end_time - serialization_start_time) * 1000} ms"
    )
    logger.info(
        f"Deserialization time: {(deserialization_end_time - deserialization_start_time) * 1000} ms"
    )


if __name__ == "__main__":
    oriented_box2d_example()
