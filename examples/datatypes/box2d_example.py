"""
Example script to demonstrate usage of Box2D datatype.
"""

import time

import numpy as np
from loguru import logger
import rerun as rr

from telekinesis import datatypes


def box2d_example():
    """
    Example function to demonstrate usage of Box2D datatype.
        - Create a Box2D data
        - Access the underlying box data
        - Visualize the Box2D data using Rerun
        - Update the Box2D data, which will update the underlying data and visualization
        - Translate the box, returns a new Box2D object with translated coordinates
        - Scale the box, returns a new Box2D object with scaled dimensions
        - Use with numpy for further operation for eg computing IoU
        - Serialize to pyarrow and back
    """
    # Create a Box2D data
    input_box2d = [1, 2.5, 3, 3]  # [x, y, width, height]
    my_box2d = datatypes.Box2D(input_box2d)
    logger.info(f"Original Box2D: {my_box2d}")

    # Access the underlying box data
    my_box2d_data = my_box2d.data
    my_box2d_shape = my_box2d.shape
    my_box2d_width = my_box2d.width
    my_box2d_height = my_box2d.height
    my_box2d_area = my_box2d.area
    my_box2d_center = my_box2d.center

    logger.info(f"Underlying Box2D data: {my_box2d_data}")
    logger.info(f"Underlying Box2D shape: {my_box2d_shape}")
    logger.info(f"Underlying Box2D width: {my_box2d_width}")
    logger.info(f"Underlying Box2D height: {my_box2d_height}")
    logger.info(f"Underlying Box2D area: {my_box2d_area}")
    logger.info(f"Underlying Box2D center: {my_box2d_center}")

    logger.info("Visualizing with Rerun...")
    rr.init("box2d_example", spawn=True)
    datatypes.visualize(my_box2d, entity_path="/Box2D/my_box2d", label="Original Box2D")

    # Update the Box2D data, which will update the underlying data and visualization
    new_box2d_data = [3, 4, 3, 5]  # New [x, y, width, height]
    my_box2d.data = new_box2d_data
    logger.info(f"Updated Box2D: {my_box2d}")
    datatypes.visualize(my_box2d, entity_path="/Box2D/my_updated_box2d", label="Updated Box2D")

    # Translate the box, returns a new Box2D object with translated coordinates
    translation_vector = [2, 3]  # [dx, dy]
    translated_box = my_box2d.translate(translation_vector)
    logger.info(f"Translated Box2D: {translated_box}")
    datatypes.visualize(
        translated_box, entity_path="/Box2D/my_translated_box2d", label="Translated Box2D"
    )

    # Scale the box, returns a new Box2D object with scaled dimensions
    scaling_factors = [2, 0.5]  # [sx, sy]
    scaled_box2d = my_box2d.scale(scaling_factors)
    logger.info(f"Scaled Box2D: {scaled_box2d}")
    datatypes.visualize(scaled_box2d, entity_path="/Box2D/my_scaled_box2d", label="Scaled Box2D")

    # Use with numpy for further operation for eg computing IoU
    my_new_box2d = my_box2d.convert_box_format(
        target_format="xyxy"
    )  # Convert to [x_min, y_min, x_max, y_max] format
    new_scaled_box2d = scaled_box2d.convert_box_format(target_format="xyxy")
    inter_min = np.maximum(my_new_box2d[:2], new_scaled_box2d.data[:2])
    inter_max = np.minimum(my_new_box2d[2:], new_scaled_box2d.data[2:])
    inter_wh = np.clip(inter_max - inter_min, 0, None)
    intersection = inter_wh[0] * inter_wh[1]
    union = my_box2d.area + scaled_box2d.area - intersection
    iou = intersection / union if union > 0 else 0.0
    logger.info(f"IoU between my_box2d and scaled_box2d: {iou}")

    # Serialize to pyarrow and back
    serialization_start_time = time.perf_counter()
    serialized = datatypes.serialize(my_box2d)
    serialization_end_time = time.perf_counter()

    deserialization_start_time = time.perf_counter()
    deserialized_box2d = datatypes.deserialize(serialized)["param_0"]
    deserialization_end_time = time.perf_counter()
    logger.info(f"Deserialized Box2D: {deserialized_box2d}")
    logger.info(f"Deserialized Box2D data matches Original: {deserialized_box2d == my_box2d}")

    logger.info(
        f"Serialization time: {(serialization_end_time - serialization_start_time) * 1000} ms"
    )
    logger.info(
        f"Deserialization time: {(deserialization_end_time - deserialization_start_time) * 1000} ms"
    )


if __name__ == "__main__":
    box2d_example()
