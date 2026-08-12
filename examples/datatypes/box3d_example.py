"""
Example script to demonstrate usage of Box3D datatype.
"""

import time

import numpy as np
from loguru import logger
import rerun as rr

from telekinesis import datatypes


def box3d_example():
    """
    Example function to demonstrate usage of Box3D datatype.
        - Create a Box3D data
        - Access the underlying box data
        - Visualize the Box3D data using Rerun
        - Update the Box3D data, which will update the underlying data and visualization
        - Translate the box, returns a new Box3D object with translated coordinates
        - Scale the box, returns a new Box3D object with scaled dimensions
        - Use with numpy for further operation for eg multiplying the box dimensions
        - Serialize to PyArrow and back
    """
    # Create a Box3D data
    input_box3d = [1, 2, 2.5, 5, 3, 5]  # [x, y, z, width, height, depth]
    my_box3d = datatypes.Box3D(input_box3d)
    logger.info(f"Original Box3D: {my_box3d}")

    # Access the underlying box data
    my_box3d_data = my_box3d.data
    my_box3d_shape = my_box3d.shape
    my_box3d_width = my_box3d.width
    my_box3d_height = my_box3d.height
    my_box3d_depth = my_box3d.depth
    my_box3d_volume = my_box3d.volume
    my_box3d_center = my_box3d.center

    logger.info(f"Underlying Box3D data: {my_box3d_data}")
    logger.info(f"Underlying Box3D shape: {my_box3d_shape}")
    logger.info(f"Underlying Box3D width: {my_box3d_width}")
    logger.info(f"Underlying Box3D height: {my_box3d_height}")
    logger.info(f"Underlying Box3D depth: {my_box3d_depth}")
    logger.info(f"Underlying Box3D volume: {my_box3d_volume}")
    logger.info(f"Underlying Box3D center: {my_box3d_center}")

    logger.info("Visualizing with Rerun...")
    rr.init("box3d_example", spawn=True)
    datatypes.visualize(my_box3d, entity_path="/Box3D/my_box3d", label="Original Box3D")

    # Update the Box3D data
    new_box3d_data = [1, 4, 2.5, 5, 2, 7]  # New [x, y, z, width, height, depth]
    my_box3d.data = new_box3d_data
    logger.info(f"Updated Box3D: {my_box3d}")
    datatypes.visualize(my_box3d, entity_path="/Box3D/my_updated_box3d", label="Updated Box3D")

    # Translate the box
    # Returns a new box3d with the translation applied, does not modify the original box3d
    translation_vector = [3, 3, 1]  # [dx, dy, dz]
    translated_box3d = my_box3d.translate(translation_vector)
    logger.info(f"Translated Box3D: {translated_box3d}")
    datatypes.visualize(
        translated_box3d, entity_path="/Box3D/my_translated_box3d", label="Translated Box3D"
    )

    # Scale the box
    # Returns a new box3d with the scaling applied, does not modify the original box3d
    scaling_factors = [2, 0.5, 1.5]  # [sx, sy, sz]
    scaled_box3d = my_box3d.scale(scaling_factors)
    logger.info(f"Scaled Box3D: {scaled_box3d}")
    datatypes.visualize(scaled_box3d, entity_path="/Box3D/my_scaled_box3d", label="Scaled Box3D")

    # Operate with Numpy on the underlying data
    multipy_factor = 1
    new_box3d_dimensions = np.multiply(my_box3d, multipy_factor)
    logger.info(
        f"Box3D dimensions multiplied by {multipy_factor} using numpy: {new_box3d_dimensions}"
    )

    # Serialize to pyarrow and back
    serialization_start_time = time.perf_counter()
    serialized = datatypes.serialize(my_box3d)
    serialization_end_time = time.perf_counter()

    deserialization_start_time = time.perf_counter()
    deserialized_box3d = datatypes.deserialize(serialized)["param_0"]
    deserialization_end_time = time.perf_counter()
    logger.info(f"Deserialized Box3D: {deserialized_box3d}")
    logger.info(f"Deserialized Box3D data matches Original: {deserialized_box3d == my_box3d}")

    logger.info(
        f"Serialization time: {(serialization_end_time - serialization_start_time) * 1000} ms"
    )
    logger.info(
        f"Deserialization time: {(deserialization_end_time - deserialization_start_time) * 1000} ms"
    )


if __name__ == "__main__":
    box3d_example()
