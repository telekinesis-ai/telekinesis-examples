"""
Example script to demonstrate usage of Contour datatype.
"""

import time

import numpy as np
from loguru import logger
import rerun as rr

from telekinesis import datatypes


def contour_example():
    """
    Example function to demonstrate usage of Contour datatype.
        - Create a Contour
        - Access the underlying points data
        - Visualize the Contour using Rerun
        - Operate on the underlying data with numpy
        - Serialize to PyArrow and back
    """
    # Create a Contour -- integer pixel coordinates, matching the (K, 2)
    # int64 array a single entry of cv2.findContours produces
    my_contour = datatypes.Contour(
        points=np.array([[118, 84], [134, 79], [150, 86], [139, 115]], dtype=np.int64),
    )
    logger.info(f"Original Contour: {my_contour}")

    # Access the underlying points data
    my_contour_points = my_contour.points
    my_contour_num_points = len(my_contour)
    logger.info(f"Underlying Contour points: {my_contour_points}")
    logger.info(f"Underlying Contour num points: {my_contour_num_points}")

    # Visualize the Contour using Rerun
    logger.info("Visualizing with Rerun...")
    rr.init("contour_example", spawn=True)
    datatypes.visualize(my_contour, entity_path="/Contour")

    # Operate on the underlying data with numpy -- shift the contour and
    # build a new Contour, since Contour is immutable after construction
    shifted_points = my_contour_points + np.array([10, 10], dtype=np.int64)
    shifted_contour = datatypes.Contour(points=shifted_points)
    logger.info(f"Shifted Contour: {shifted_contour}")
    datatypes.visualize(shifted_contour, entity_path="/Contour/shifted")

    # Serialize to PyArrow and back
    serialization_start_time = time.perf_counter()
    serialized = datatypes.serialize(shifted_contour)
    serialization_end_time = time.perf_counter()

    deserialization_start_time = time.perf_counter()
    deserialized = datatypes.deserialize(serialized)["param_0"]
    deserialization_end_time = time.perf_counter()

    logger.info(f"Deserialized Contour matches Original: {deserialized == shifted_contour}")
    logger.info(
        f"Serialization time: {(serialization_end_time - serialization_start_time) * 1000} ms"
    )
    logger.info(
        f"Deserialization time: {(deserialization_end_time - deserialization_start_time) * 1000} ms"
    )


if __name__ == "__main__":
    contour_example()
