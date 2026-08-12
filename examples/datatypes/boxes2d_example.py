"""
Example script to demonstrate usage of Boxes2D datatype.
"""

import time

import numpy as np
from loguru import logger
import rerun as rr

from telekinesis import datatypes


def boxes2d_example():
    """
    Example function to demonstrate usage of Boxes2D datatype.
     - Create a Boxes2D data
     - Print the original data
    """
    # Create a Boxes2D data
    input_box2d_1 = [[1, 2.5, 3, 3]]  # [[x, y, width, height], ...]
    input_box2d_2 = [[4, 5, 2, 1]]  # [[x, y, width, height], ...]
    input_boxes2d = np.concatenate([input_box2d_1, input_box2d_2], axis=0)
    my_boxes2d = datatypes.Boxes2D(input_boxes2d)
    logger.info(f"Original Boxes2D: {my_boxes2d}")

    # Access the underlying box data
    my_box2d_data = my_boxes2d.data
    my_box2d_shape = my_boxes2d.shape
    my_box2d_width = my_boxes2d.width
    my_box2d_height = my_boxes2d.height
    my_box2d_area = my_boxes2d.area
    my_box2d_center = my_boxes2d.center

    logger.info(f"Underlying Box2D data: {my_box2d_data}")
    logger.info(f"Underlying Box2D shape: {my_box2d_shape}")
    logger.info(f"Underlying Box2D width: {my_box2d_width}")
    logger.info(f"Underlying Box2D height: {my_box2d_height}")
    logger.info(f"Underlying Box2D area: {my_box2d_area}")
    logger.info(f"Underlying Box2D center: {my_box2d_center}")

    logger.info("Visualizing with Rerun...")
    rr.init("boxes2d_example", spawn=True)
    datatypes.visualize(
        my_boxes2d, entity_path="/Boxes2D/my_box2d", label=["Original Box2D 1", "Original Box2D 2"]
    )

    # Update box index 1 in place, keeping box 0 unchanged
    new_box2d_data = [3, 4, 3, 5]  # New [x, y, width, height]
    updated_data = my_boxes2d.data
    updated_data[1] = new_box2d_data
    my_boxes2d.data = updated_data
    logger.info(f"Updated Box2D: {my_boxes2d}")
    datatypes.visualize(
        my_boxes2d,
        entity_path="/Boxes2D/my_updated_box2d",
        label=["Original Box2D 1", "Original Box2D 2"],
    )

    # Translate all the boxes, returns a new Boxes2D object with translated coordinates
    translation_vector = [2, 3]  # [dx, dy]
    translated_boxes2d = my_boxes2d.translate(translation_vector)
    logger.info(f"Translated Boxes2D: {translated_boxes2d}")
    datatypes.visualize(
        translated_boxes2d,
        entity_path="/Boxes2D/my_translated_box2d",
        label=["Translated Box2D 1", "Translated Box2D 2"],
    )

    # Scale all the boxes, returns a new Boxes2D object with scaled dimensions
    scaling_factors = [0.5, 0.5]  # [sx, sy]
    scaled_boxes2d = my_boxes2d.scale(scaling_factors)
    logger.info(f"Scaled Boxes2D: {scaled_boxes2d}")
    datatypes.visualize(
        scaled_boxes2d,
        entity_path="/Boxes2D/my_scaled_box2d",
        label=["Scaled Box2D 1", "Scaled Box2D 2"],
    )

    # Serialize to pyarrow and back
    serialization_start_time = time.perf_counter()
    serialized = datatypes.serialize(my_boxes2d)
    serialization_end_time = time.perf_counter()

    deserialization_start_time = time.perf_counter()
    deserialized_box2d = datatypes.deserialize(serialized)["param_0"]
    deserialization_end_time = time.perf_counter()
    logger.info(f"Deserialized Box2D: {deserialized_box2d}")
    logger.info(f"Deserialized Box2D data matches Original: {deserialized_box2d == my_boxes2d}")

    logger.info(
        f"Serialization time: {(serialization_end_time - serialization_start_time) * 1000} ms"
    )
    logger.info(
        f"Deserialization time: {(deserialization_end_time - deserialization_start_time) * 1000} ms"
    )


if __name__ == "__main__":
    boxes2d_example()
