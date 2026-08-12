"""
Example script to demonstrate usage of Boxes3D datatype.
"""

import time

import numpy as np
from loguru import logger
import rerun as rr

from telekinesis import datatypes


def boxes3d_example():
    """
    Example function to demonstrate usage of Boxes3D datatype.
     - Create a Boxes3D data
     - Print the original data
    """
    # Create a Boxes3D data
    input_box3d_1 = [[0, 0, 0, 1, 1, 1]]  # [[x, y, z, width, height, depth], ...]
    input_box3d_2 = [[2, 2, 2, 3, 3, 3]]  # [[x, y, z, width, height, depth], ...]
    input_boxes3d = np.concatenate([input_box3d_1, input_box3d_2], axis=0)
    my_boxes3d = datatypes.Boxes3D(input_boxes3d)
    logger.info(f"Original Boxes3D: {my_boxes3d}")

    # Access the underlying box data
    my_box3d_data = my_boxes3d.data
    my_box3d_shape = my_boxes3d.shape
    my_box3d_width = my_boxes3d.width
    my_box3d_height = my_boxes3d.height
    my_box3d_depth = my_boxes3d.depth
    my_box3d_volume = my_boxes3d.volume
    my_box3d_center = my_boxes3d.center

    logger.info(f"Underlying Box3D data: {my_box3d_data}")
    logger.info(f"Underlying Box3D shape: {my_box3d_shape}")
    logger.info(f"Underlying Box3D width: {my_box3d_width}")
    logger.info(f"Underlying Box3D height: {my_box3d_height}")
    logger.info(f"Underlying Box3D depth: {my_box3d_depth}")
    logger.info(f"Underlying Box3D volume: {my_box3d_volume}")
    logger.info(f"Underlying Box3D center: {my_box3d_center}")

    logger.info("Visualizing with Rerun...")
    rr.init("boxes3d_example", spawn=True)
    datatypes.visualize(
        my_boxes3d, entity_path="/Boxes3D/my_box3d", label=["Original Box3D 1", "Original Box3D 2"]
    )

    # Update box index 1 in place, keeping box 0 unchanged
    new_box3d_data = [3, 3, 3, 1, 1, 1]  # New [x, y, z, width, height, depth]
    updated_data = my_boxes3d.data
    updated_data[1] = new_box3d_data
    my_boxes3d.data = updated_data
    logger.info(f"Updated Box3D: {my_boxes3d}")
    datatypes.visualize(
        my_boxes3d,
        entity_path="/Boxes3D/my_updated_box3d",
        label=["Original Box3D 1", "Original Box3D 2"],
    )

    # Translate all the boxes, returns a new Boxes3D object with translated coordinates
    translation_vector = [2, 3, 1]  # [dx, dy, dz]
    translated_boxes3d = my_boxes3d.translate(translation_vector)
    logger.info(f"Translated Boxes3D: {translated_boxes3d}")
    datatypes.visualize(
        translated_boxes3d,
        entity_path="/Boxes3D/my_translated_box3d",
        label=["Translated Box3D 1", "Translated Box3D 2"],
    )

    # Scale all the boxes, returns a new Boxes3D object with scaled dimensions
    scaling_factors = [0.5, 0.5, 0.5]  # [sx, sy, sz]
    scaled_boxes3d = my_boxes3d.scale(scaling_factors)
    logger.info(f"Scaled Boxes3D: {scaled_boxes3d}")
    datatypes.visualize(
        scaled_boxes3d,
        entity_path="/Boxes3D/my_scaled_box3d",
        label=["Scaled Box3D 1", "Scaled Box3D 2"],
    )

    # Serialize to pyarrow and back
    serialization_start_time = time.perf_counter()
    serialized = datatypes.serialize(my_boxes3d)
    serialization_end_time = time.perf_counter()

    deserialization_start_time = time.perf_counter()
    deserialized_box3d = datatypes.deserialize(serialized)["param_0"]
    deserialization_end_time = time.perf_counter()
    logger.info(f"Deserialized Box3D: {deserialized_box3d}")
    logger.info(f"Deserialized Box3D data matches Original: {deserialized_box3d == my_boxes3d}")

    logger.info(
        f"Serialization time: {(serialization_end_time - serialization_start_time) * 1000} ms"
    )
    logger.info(
        f"Deserialization time: {(deserialization_end_time - deserialization_start_time) * 1000} ms"
    )


if __name__ == "__main__":
    boxes3d_example()
