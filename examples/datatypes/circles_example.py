"""
Example script to demonstrate usage of Circles datatype.
"""

import time

import numpy as np
from loguru import logger
import rerun as rr

from telekinesis import datatypes


def circles_example():
    """
    Example function to demonstrate usage of Circles datatype.
        - Create a Circles data
        - Access the underlying data
        - Visualize the Circles data using Rerun
        - Index a single circle
        - Translate and scale the batch, each returning a new Circles
        - Serialize to PyArrow and back
    """
    # Create a Circles data
    my_circles = datatypes.Circles(
        centers=np.array([[50.0, 60.0], [120.0, 80.0], [200.0, 150.0]], dtype=np.float32),
        radii=np.array([10.0, 15.5, 7.25], dtype=np.float32),
    )
    logger.info(f"Original Circles: {my_circles}")

    # Access the underlying annotations data
    my_circle_centers = my_circles.centers
    my_circle_radii = my_circles.radii
    logger.info(f"Underlying Circles centers data: {my_circle_centers}")
    logger.info(f"Underlying Circles radii data: {my_circle_radii}")

    # Visualize the Circles data using Rerun
    logger.info("Visualizing with Rerun...")
    rr.init("circles_example", spawn=True)
    datatypes.visualize(
        my_circles,
        entity_path="/Circles",
        label=["My Circle 1", "My Circle 2", "My Circle 3"],
    )

    # Index a single circle (returns a Circle) and
    first_circle = my_circles[0]
    logger.info(f"First circle: center={first_circle.center}, radius={first_circle.radius}")

    # Index a sub-batch of circles (returns a Circles)
    sub_batch = my_circles[1:]
    logger.info(f"Sub-batch [1:]: centers={sub_batch.centers}, radii={sub_batch.radii}")

    # Translate and scale the batch, each returning a new Circles
    my_circles = my_circles.translate(
        [[5.0, 5.0], [10.0, 0.0], [-5.0, -5.0]]
    ).scale(1.5)
    logger.info(f"Updated Circles centers data: {my_circles.centers}")
    logger.info(f"Updated Circles radii data: {my_circles.radii}")
    datatypes.visualize(
        my_circles,
        entity_path="/Circles/updated",
        label=["Updated Circle 1", "Updated Circle 2", "Updated Circle 3"],
    )

    # Create an empty Circles instance
    empty_centers = np.empty((0, 2), dtype=np.float32)
    empty_radii = np.empty((0,), dtype=np.float32)
    empty_circles = datatypes.Circles(centers=empty_centers, radii=empty_radii)
    logger.info(f"My Empty Circles: {empty_circles}")

    # Serialize to PyArrow and back
    serialization_start_time = time.perf_counter()
    serialized = datatypes.serialize(my_circles)
    serialization_end_time = time.perf_counter()

    deserialization_start_time = time.perf_counter()
    deserialized = datatypes.deserialize(serialized)["param_0"]
    deserialization_end_time = time.perf_counter()

    logger.info(
        f"Deserialized Circles matches Original: {deserialized == my_circles}"
    )
    logger.info(
        f"Serialization time: {(serialization_end_time - serialization_start_time) * 1000} ms"
    )
    logger.info(
        f"Deserialization time: {(deserialization_end_time - deserialization_start_time) * 1000} ms"
    )


if __name__ == "__main__":
    circles_example()
