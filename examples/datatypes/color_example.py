"""
Example script to demonstrate usage of Color datatype.
"""

import time

from loguru import logger
import rerun as rr

from telekinesis import datatypes


def rgba32_example():
    """
    Example function to demonstrate usage of Color datatype.
     - Create an Color object with RGBA values
     - Access the underlying color attributes
     - Visualize the Color data using Rerun
     - Update the Color data and visualize again
    """
    # Create an Rgba32 data
    input_color = [255, 0, 128]
    my_color = datatypes.Color(input_color)
    logger.info(f"Original Color: {my_color}")

    # Access the underlying color attributes
    my_color_data = my_color.data
    my_color_shape = my_color.shape
    my_color_size = my_color.size
    my_color_dtype = my_color.dtype
    my_color_ndim = my_color.ndim
    my_color_numpy = my_color.to_numpy()
    my_color_copy = my_color.copy()

    logger.info(f"Underlying Color shape: {my_color_shape}")
    logger.info(f"Underlying Color size: {my_color_size}")
    logger.info(f"Underlying Color dtype: {my_color_dtype}")
    logger.info(f"Underlying Color ndim: {my_color_ndim}")
    logger.info(f"Underlying Color as numpy array: {my_color_numpy}")
    logger.info(f"Underlying Color copy: {my_color_copy}")

    # Visualize the Color data using Rerun
    logger.info("Visualizing with Rerun...")
    rr.init("rgba32_example", spawn=True)
    datatypes.visualize(my_color, entity_path="/Color")

    # Update the my_color_data
    new_color_data = [0, 255, 255, 255]
    my_color.data = new_color_data
    logger.info(f"Updated Color: {my_color}")
    datatypes.visualize(my_color, entity_path="/Color/updated")

    # Create color from hex string
    hex_color = "#FF00FF80"
    color_from_hex = datatypes.Color.from_hex(hex_color)
    logger.info(f"Color from hex {hex_color}: {color_from_hex}")
    datatypes.visualize(color_from_hex, entity_path="/Color/from_hex")

    # From the hex string, convert back to hex to verify
    hex_from_color = color_from_hex.to_hex()
    logger.info(f"Hex from Color: {hex_from_color}")
    logger.info(f"Hex conversion successful: {hex_from_color == hex_color}")

    # Serialize to PyArrow and back
    serialization_start_time = time.perf_counter()
    serialized = datatypes.serialize(my_color)
    serialization_end_time = time.perf_counter()

    deserialization_start_time = time.perf_counter()
    deserialized = datatypes.deserialize(serialized)["param_0"]
    deserialization_end_time = time.perf_counter()
    logger.info(f"Deserialized Color: {deserialized}")
    logger.info(f"Deserialized and original Color are equal: {deserialized == my_color}")

    logger.info(f"Serialization time: {serialization_end_time - serialization_start_time} seconds")
    logger.info(
        f"Deserialization time: {deserialization_end_time - deserialization_start_time} seconds"
    )


if __name__ == "__main__":
    rgba32_example()
