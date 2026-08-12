"""
Example script to demonstrate usage of OrientedBox3D datatype.
"""

import time

from loguru import logger
import rerun as rr

from telekinesis import datatypes


def oriented_box3d_example():
    """
    Example function to demonstrate usage of OrientedBox3D datatype.
        - Create an OrientedBox3D data
        - Access the underlying box data
        - Visualize the OrientedBox3D data using Rerun
        - Update the underlying box data
        - Translate the box, returns a new OrientedBox3D object with updated center
        - Scale the box, returns a new OrientedBox3D object with updated size
        - Rotate the box, returns a new OrientedBox3D object with updated quaternion
        - Serialize to PyArrow and back
    """
    # Create an OrientedBox3D data: [x, y, z, w, h, l, qx, qy, qz, qw], identity rotation
    my_oriented_box3d = datatypes.OrientedBox3D([0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0])
    logger.info(f"Original OrientedBox3D: {my_oriented_box3d}")

    # Access the underlying box data
    my_oriented_box3d_data = my_oriented_box3d.data
    my_oriented_box3d_shape = my_oriented_box3d.shape
    my_oriented_box3d_dtype = my_oriented_box3d.dtype
    my_oriented_box3d_ndim = my_oriented_box3d.ndim
    my_oriented_box3d_numpy = my_oriented_box3d.to_numpy()
    my_oriented_box3d_center = my_oriented_box3d.center
    my_oriented_box3d_volume = my_oriented_box3d.volume
    my_oriented_box3d_width = my_oriented_box3d.width
    my_oriented_box3d_height = my_oriented_box3d.height
    my_oriented_box3d_depth = my_oriented_box3d.depth
    my_oriented_box3d_quaternion = my_oriented_box3d.data[6:]  # [qx, qy, qz, qw]

    logger.info("Visualizing with Rerun...")
    rr.init("oriented_box3d_example", spawn=True)
    datatypes.visualize(
        my_oriented_box3d, entity_path="/OrientedBox3D/my_oriented_box3d", label="My Oriented Box3D"
    )

    logger.info(f"Underlying OrientedBox3D data: {my_oriented_box3d_data}")
    logger.info(f"Underlying OrientedBox3D data shape: {my_oriented_box3d_shape}")
    logger.info(f"Underlying OrientedBox3D data dtype: {my_oriented_box3d_dtype}")
    logger.info(f"Underlying OrientedBox3D data ndim: {my_oriented_box3d_ndim}")
    logger.info(f"Underlying OrientedBox3D data as numpy array: {my_oriented_box3d_numpy}")
    logger.info(f"Underlying OrientedBox3D center: {my_oriented_box3d_center}")
    logger.info(f"Underlying OrientedBox3D volume: {my_oriented_box3d_volume}")
    logger.info(f"Underlying OrientedBox3D width: {my_oriented_box3d_width}")
    logger.info(f"Underlying OrientedBox3D height: {my_oriented_box3d_height}")
    logger.info(f"Underlying OrientedBox3D depth: {my_oriented_box3d_depth}")
    logger.info(
        f"Underlying OrientedBox3D quaternion [qx, qy, qz, qw]: {my_oriented_box3d_quaternion}"
    )

    # Update the my_oriented_box3d data -- markedly longer along x than y/z, so a
    # 90-degree rotation about z is unmistakable in the viewer
    my_oriented_box3d.data = [2.0, 2.0, 2.0, 3.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    logger.info(f"Updated OrientedBox3D: {my_oriented_box3d}")
    datatypes.visualize(
        my_oriented_box3d,
        entity_path="/OrientedBox3D/my_updated_oriented_box3d",
        label="Updated Oriented Box3D",
    )

    # Translate the box, returns a new OrientedBox3D object with updated center
    translated_oriented_box3d = my_oriented_box3d.translate([1.0, 1.0, 1.0])
    logger.info(
        f"Translated OrientedBox3D center: {translated_oriented_box3d.center} "
        f"(was {my_oriented_box3d.center})"
    )
    datatypes.visualize(
        translated_oriented_box3d,
        entity_path="/OrientedBox3D/my_translated_oriented_box3d",
        label="Translated Oriented Box3D",
    )

    # Scale the box, returns a new OrientedBox3D object with updated size
    scaled_oriented_box3d = my_oriented_box3d.scale(1.5)
    logger.info(
        f"Scaled OrientedBox3D width, height, and depth: {scaled_oriented_box3d.width} x {scaled_oriented_box3d.height} x {scaled_oriented_box3d.depth} "
        f"(was {my_oriented_box3d.width} x {my_oriented_box3d.height} x {my_oriented_box3d.depth})"
    )
    datatypes.visualize(
        scaled_oriented_box3d,
        entity_path="/OrientedBox3D/my_scaled_oriented_box3d",
        label="Scaled Oriented Box3D",
    )

    # Rotate the box by a 90-degree turn about Z, returns a new OrientedBox3D object
    # with the composed quaternion (rotation is [qx, qy, qz, qw], this class's quat_order)
    delta_quaternion = [0.0, 0.0, 0.70710678, 0.70710678]  # 90 degrees about Z
    rotated_oriented_box3d = my_oriented_box3d.rotate(delta_quaternion)
    logger.info(
        f"Rotated OrientedBox3D quaternion: {rotated_oriented_box3d.data[6:]} "
        f"(was {my_oriented_box3d.data[6:]})"
    )
    # rotate() keeps the box's center in place, so it would otherwise sit exactly on
    # top of my_updated_oriented_box3d -- translate it aside purely for visualization,
    # so the reoriented long axis (now along y instead of x) is easy to see on its own
    rotated_oriented_box3d_display = rotated_oriented_box3d.translate([4.0, 0.0, 0.0])
    datatypes.visualize(
        rotated_oriented_box3d_display,
        entity_path="/OrientedBox3D/my_rotated_oriented_box3d",
        label="Rotated Oriented Box3D",
    )

    # Serialize to PyArrow and back
    serialization_start_time = time.perf_counter()
    serialized = datatypes.serialize(my_oriented_box3d)
    serialization_end_time = time.perf_counter()

    deserialization_start_time = time.perf_counter()
    deserialized = datatypes.deserialize(serialized)["param_0"]
    deserialization_end_time = time.perf_counter()

    logger.info(f"Deserialized OrientedBox3D matches Original: {deserialized == my_oriented_box3d}")
    logger.info(
        f"Serialization time: {(serialization_end_time - serialization_start_time) * 1000} ms"
    )
    logger.info(
        f"Deserialization time: {(deserialization_end_time - deserialization_start_time) * 1000} ms"
    )


if __name__ == "__main__":
    oriented_box3d_example()
