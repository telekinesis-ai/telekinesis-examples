"""
Example script to demonstrate usage of Circle datatype.
"""

import time

import numpy as np
from loguru import logger
import rerun as rr

from telekinesis import datatypes


def circle_example():
    """
    Example function to demonstrate usage of Circle datatype.
        - Create a Circle
        - Access the underlying center/radius data
        - Translate and scale the circle, each returning a new Circle
        - Visualize the Circle using Rerun
        - Serialize to PyArrow and back
    """
    # Create a Circle
    my_circle = datatypes.Circle(center=[50.0, 60.0], radius=10.0)
    logger.info(f"Original Circle: {my_circle}")

    # Access the underlying center/radius data
    my_circle_center = my_circle.center
    my_circle_radius = my_circle.radius
    logger.info(f"Underlying Circle center: {my_circle_center}")
    logger.info(f"Underlying Circle radius: {my_circle_radius}")

    # Circle is immutable after construction (no `center`/`radius` setter) --
    # translate/scale each return a new instance instead.
    translated_circle = my_circle.translate([5.0, 5.0])
    scaled_circle = translated_circle.scale(1.5)
    logger.info(f"Translated Circle: {translated_circle}")
    logger.info(f"Scaled Circle: {scaled_circle}")

    # Visualize the Circle using Rerun
    logger.info("Visualizing with Rerun...")
    rr.init("circle_example", spawn=True)
    datatypes.visualize(my_circle, entity_path="/Circle", label="My Circle")
    datatypes.visualize(scaled_circle, entity_path="/Circle/updated", label="Updated Circle")

    # Serialize to PyArrow and back
    serialization_start_time = time.perf_counter()
    serialized = datatypes.serialize(scaled_circle)
    serialization_end_time = time.perf_counter()

    deserialization_start_time = time.perf_counter()
    deserialized = datatypes.deserialize(serialized)["param_0"]
    deserialization_end_time = time.perf_counter()

    logger.info(f"Deserialized Circle matches Original: {deserialized == scaled_circle}")
    logger.info(
        f"Serialization time: {(serialization_end_time - serialization_start_time) * 1000} ms"
    )
    logger.info(
        f"Deserialization time: {(deserialization_end_time - deserialization_start_time) * 1000} ms"
    )


if __name__ == "__main__":
    circle_example()
