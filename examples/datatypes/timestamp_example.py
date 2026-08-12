"""
Example script to demonstrate usage of Twist3D datatype.
"""

import time
from datetime import datetime, timezone

from loguru import logger
import rerun as rr

from telekinesis import datatypes


def timestamp_example():
    """
    Example function to demonstrate usage of Twist3D datatype.
     - Create a Twist3D data
     - Print the original data
    """
    # Create a Timestamp data
    input_timestamp = datetime.now(timezone.utc)
    my_timestamp = datatypes.Timestamp(input_timestamp)
    logger.info(f"Original Timestamp: {my_timestamp}")

    # Access the underlying twist data
    my_timestamp_data = my_timestamp.data
    logger.info(f"Underlying Timestamp data: {my_timestamp_data}")
    logger.info("Visualizing with Rerun...")
    rr.init("timestamp_example", spawn=True)
    datatypes.visualize(my_timestamp, entity_path="/Timestamp/my_timestamp")

    # Update the timestamp data
    updated_timestamp = datetime.now(timezone.utc)
    my_timestamp.data = updated_timestamp
    logger.info(f"Updated Timestamp: {my_timestamp}")
    datatypes.visualize(my_timestamp, entity_path="/Timestamp/updated", label="Updated Timestamp")

    # Difference between the original and updated timestamp
    time_difference = my_timestamp.data - my_timestamp_data
    logger.info(f"Time difference between original and updated timestamp: {time_difference}")

    # Serialize to PyArrow and back
    serialization_start_time = time.perf_counter()
    serialized_timestamp = datatypes.serialize(my_timestamp)
    serialization_end_time = time.perf_counter()

    deserialization_start_time = time.perf_counter()
    deserialized_timestamp = datatypes.deserialize(serialized_timestamp)
    deserialization_end_time = time.perf_counter()
    logger.info(f"Deserialized Timestamp: {deserialized_timestamp}")

    logger.info(
        f"Serialization time: {(serialization_end_time - serialization_start_time) * 1000} ms"
    )
    logger.info(
        f"Deserialization time: {(deserialization_end_time - deserialization_start_time) * 1000} ms"
    )


if __name__ == "__main__":
    timestamp_example()
