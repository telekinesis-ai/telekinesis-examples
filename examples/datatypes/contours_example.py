"""
Example script to demonstrate usage of Contours datatype.
"""

import time

import numpy as np
from loguru import logger
import rerun as rr

from telekinesis import datatypes


def contour_annotations_example():
    """
    Example function to demonstrate usage of Contours datatype.
        - Create a Contours data
        - Access the underlying annotations data
        - Visualize the Contours data using Rerun
        - Update the underlying annotations data
        - Operate on the underlying data with numpy
        - Serialize to PyArrow and back
    """
    # Create a Contours data -- integer pixel coordinates, matching
    # the (K_i, 2) int64 arrays cv2.findContours produces
    input_countour_1 = np.array([[118, 84], [134, 79], [150, 86], [139, 115]], dtype=np.int64)
    input_countour_2 = np.array([[210, 145], [238, 140], [276, 160], [245, 190]], dtype=np.int64)
    input_countour_3 = np.array([[322, 212], [335, 211], [332, 225], [320, 218]], dtype=np.int64)
    input_contours = [input_countour_1, input_countour_2, input_countour_3]
    my_contour_annotations = datatypes.Contours(
        data=input_contours,
    )
    logger.info(f"Original Contours: {my_contour_annotations}")

    # Access the underlying annotations data
    my_contour_annotations_data = my_contour_annotations.data
    my_contour_annotations_contour_1 = my_contour_annotations_data[0]
    my_contour_annotations_lengths = len(my_contour_annotations)
    logger.info(f"Underlying Contours data: {my_contour_annotations_data}")
    logger.info(f"Underlying Contours contour 1: {my_contour_annotations_contour_1}")
    logger.info(f"Underlying Contours lengths: {my_contour_annotations_lengths}")

    logger.info("Visualizing with Rerun...")
    rr.init("contour_annotations_example", spawn=True)
    datatypes.visualize(my_contour_annotations, entity_path="/Contours")

    # Create empty contours
    input_contours_1_empty = np.empty((0, 2), dtype=np.int64)
    input_contours_empty = [input_contours_1_empty]
    my_contour_annotations_empty = datatypes.Contours(
        data=input_contours_empty,
    )
    logger.info(f"Empty Contours: {my_contour_annotations_empty}")

    # Serialization to PyArrow and back
    serialization_start_time = time.perf_counter()
    serialized = datatypes.serialize(my_contour_annotations)
    serialization_end_time = time.perf_counter()

    deserialization_start_time = time.perf_counter()
    deserialized = datatypes.deserialize(serialized)["param_0"]
    deserialization_end_time = time.perf_counter()
    logger.info(f"Deserialized Contours matches original: {deserialized == my_contour_annotations}")

    logger.info(
        f"Serialization took {(serialization_end_time - serialization_start_time) * 1000} ms."
    )
    logger.info(
        f"Deserialization took {(deserialization_end_time - deserialization_start_time) * 1000} ms."
    )


if __name__ == "__main__":
    contour_annotations_example()
