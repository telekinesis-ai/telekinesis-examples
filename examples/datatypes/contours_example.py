"""
Example script to demonstrate usage of Contours datatype.
"""

import time

import rerun as rr
import numpy as np
from loguru import logger

from telekinesis import datatypes


def contours_example():
    """
    Example function to demonstrate usage of Contours datatype.
        - Create a Contours object
        - Access the underlying contours data
        - Visualize the Contours data using Rerun
        - Update the underlying contours data
        - Operate on the underlying data with numpy
        - Serialize to PyArrow and back
    """
    # Create a Contours data -- integer pixel coordinates, matching
    # the (K_i, 2) int64 arrays cv2.findContours produces
    input_contour_1 = np.array([[118, 84], [134, 79], [150, 86], [139, 115]], dtype=np.int64)
    input_contour_2 = np.array([[210, 145], [238, 140], [276, 160], [245, 190]], dtype=np.int64)
    input_contour_3 = np.array([[322, 212], [335, 211], [332, 225], [320, 218]], dtype=np.int64)
    input_contours = [input_contour_1, input_contour_2, input_contour_3]
    my_contours = datatypes.Contours(
        points=input_contours,
    )
    logger.info(f"Original Contours: {my_contours}")

    # Access the underlying contours data
    my_contour_data = my_contours.points
    my_contour_1 = my_contour_data[0]
    my_contour_lengths = len(my_contours)
    logger.info(f"Underlying Contours data: {my_contour_data}")
    logger.info(f"Underlying Contours contour 1: {my_contour_1}")
    logger.info(f"Underlying Contours lengths: {my_contour_lengths}")

    logger.info("Visualizing with Rerun...")
    rr.init("contours_example", spawn=True)
    datatypes.visualize(my_contours, entity_path="/Contours")

    # Access the first contour and log its details
    first_contour = my_contours[0]
    logger.info(f"First contour: {first_contour}")

    # Access a sub-batch of contours and log its details
    sub_batch = my_contours[1:]
    logger.info(f"Sub-batch of contours [1:]: {sub_batch}")

    # Create empty contours
    input_contours_1_empty = np.empty((0, 2), dtype=np.int64)
    input_contours_empty = [input_contours_1_empty]
    my_contours_empty = datatypes.Contours(
        points=input_contours_empty,
    )
    logger.info(f"Empty Contours: {my_contours_empty}")

    # Serialization to PyArrow and back
    serialization_start_time = time.perf_counter()
    serialized = datatypes.serialize(my_contours)
    serialization_end_time = time.perf_counter()

    deserialization_start_time = time.perf_counter()
    deserialized = datatypes.deserialize(serialized)["param_0"]
    deserialization_end_time = time.perf_counter()
    logger.info(f"Deserialized Contours matches original: {deserialized == my_contours}")

    logger.info(
        f"Serialization took {(serialization_end_time - serialization_start_time) * 1000} ms."
    )
    logger.info(
        f"Deserialization took {(deserialization_end_time - deserialization_start_time) * 1000} ms."
    )


if __name__ == "__main__":
    contours_example()
