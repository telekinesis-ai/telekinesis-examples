"""
Example script to demonstrate usage of Circles datatype.
"""

import time

import numpy as np
from loguru import logger
import rerun as rr

from telekinesis import datatypes


def circle_detection_annotations_example():
    """
    Example function to demonstrate usage of Circles datatype.
        - Create a Circles data
        - Access the underlying annotations data
        - Visualize the Circles data using Rerun
        - Index a single circle and a sub-batch with __getitem__
        - Translate and scale the batch, each returning a new Circles
        - Serialize to PyArrow and back
    """
    # Create a Circles data
    my_circle_detection_annotations = datatypes.Circles(
        centers=np.array([[50.0, 60.0], [120.0, 80.0], [200.0, 150.0]], dtype=np.float32),
        radii=np.array([10.0, 15.5, 7.25], dtype=np.float32),
    )
    logger.info(f"Original Circles: {my_circle_detection_annotations}")

    # Access the underlying annotations data
    my_circle_detection_annotations_centers = my_circle_detection_annotations.centers
    my_circle_detection_annotations_radii = my_circle_detection_annotations.radii
    logger.info(f"Underlying Circles centers data: {my_circle_detection_annotations_centers}")
    logger.info(f"Underlying Circles radii data: {my_circle_detection_annotations_radii}")

    # Visualize the Circles data using Rerun
    logger.info("Visualizing with Rerun...")
    rr.init("circle_detection_example", spawn=True)
    datatypes.visualize(
        my_circle_detection_annotations,
        entity_path="/Circles",
        label=["My Circle 1", "My Circle 2", "My Circle 3"],
    )

    # Index a single circle (returns a length-1 Circles) and a sub-batch with __getitem__
    first_circle = my_circle_detection_annotations[0]
    logger.info(f"First circle: center={first_circle.centers[0]}, radius={first_circle.radii[0]}")
    sub_batch = my_circle_detection_annotations[1:]
    logger.info(f"Sub-batch [1:]: centers={sub_batch.centers}, radii={sub_batch.radii}")

    # Circles is immutable after construction (no `centers`/`radii`
    # setter) -- translate/scale each return a new instance instead.
    my_circle_detection_annotations = my_circle_detection_annotations.translate(
        [[5.0, 5.0], [10.0, 0.0], [-5.0, -5.0]]
    ).scale(1.5)
    logger.info(f"Updated Circles centers data: {my_circle_detection_annotations.centers}")
    logger.info(f"Updated Circles radii data: {my_circle_detection_annotations.radii}")
    datatypes.visualize(
        my_circle_detection_annotations,
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
    serialized = datatypes.serialize(my_circle_detection_annotations)
    serialization_end_time = time.perf_counter()

    deserialization_start_time = time.perf_counter()
    deserialized = datatypes.deserialize(serialized)["param_0"]
    deserialization_end_time = time.perf_counter()

    logger.info(
        f"Deserialized Circles matches Original: {deserialized == my_circle_detection_annotations}"
    )
    logger.info(
        f"Serialization time: {(serialization_end_time - serialization_start_time) * 1000} ms"
    )
    logger.info(
        f"Deserialization time: {(deserialization_end_time - deserialization_start_time) * 1000} ms"
    )


if __name__ == "__main__":
    circle_detection_annotations_example()
