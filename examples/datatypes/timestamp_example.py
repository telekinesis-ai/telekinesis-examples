"""Demonstrates the Telekinesis Timestamp datatype."""

import time
from datetime import datetime, timezone

from loguru import logger
import rerun as rr

from telekinesis import datatypes


def timestamp_example():
    """Demonstrate creation, access, visualization, update, arithmetic, and serialization."""

    # ======================= Create ============================================
    timestamp = datatypes.Timestamp(datetime.now(timezone.utc))
    logger.info(f"Original Timestamp: {timestamp}")

    # ======================= Inspect ===========================================
    data = timestamp.data
    logger.info(f"Underlying Timestamp data: {data}")

    # ======================= Visualize =========================================
    rr.init("timestamp_example", spawn=True)
    datatypes.visualize(timestamp, entity_path="/Timestamp/my_timestamp")

    # ======================= Update ============================================
    timestamp.data = datetime.now(timezone.utc)
    logger.info(f"Updated Timestamp: {timestamp}")
    datatypes.visualize(timestamp, entity_path="/Timestamp/updated", label="Updated Timestamp")

    # ======================= Arithmetic ========================================
    diff = timestamp.data - data
    logger.info(f"Time difference between original and updated timestamp: {diff}")

    # ======================= Serialize / Deserialize ===========================
    start = time.perf_counter()
    serialized = datatypes.serialize(timestamp)
    serialization_ms = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    deserialized = datatypes.deserialize(serialized)["param_0"]
    deserialization_ms = (time.perf_counter() - start) * 1000

    logger.info(f"Deserialized Timestamp: {deserialized}")
    logger.info(f"Round-trip successful: {timestamp == deserialized}")
    logger.info(f"Serialization time: {serialization_ms:.3f} ms")
    logger.info(f"Deserialization time: {deserialization_ms:.3f} ms")


if __name__ == "__main__":
    timestamp_example()
