"""Demonstrates the Telekinesis Timestamp datatype."""

import time

import rerun as rr
from loguru import logger

from telekinesis import datatypes

def timestamp_example():
    """Demonstrate creation, inspection, operations, visualization, and serialization."""

    # ======================= Create ============================================
    timestamp = datatypes.Timestamp(sec=42, nanosec=250_000_000)
    logger.info(f"Created Timestamp: {timestamp}")

    # ======================= Inspect ===========================================
    logger.info(f"sec={timestamp.sec}")
    logger.info(f"nanosec={timestamp.nanosec}")

    # ======================= Operations ========================================
    # Timestamp is immutable; build a new instance rather than mutating in place.
    later = datatypes.Timestamp(sec=43, nanosec=0)
    logger.info(f"Later Timestamp: {later}")

    same = datatypes.Timestamp(sec=42, nanosec=250_000_000)
    logger.info(f"EQ: {timestamp} == {same} = {timestamp == same}")
    logger.info(f"EQ: {timestamp} == {later} = {timestamp == later}")

    # Timestamp exposes only sec/nanosec and equality; derived quantities such
    # as elapsed time are computed by the caller from the raw fields.
    diff_sec = (later.sec + later.nanosec / 1e9) - (timestamp.sec + timestamp.nanosec / 1e9)
    logger.info(f"Difference between timestamps: {diff_sec:.3f} s")

    # ======================= Visualize =========================================
    rr.init("timestamp_example", spawn=True)
    datatypes.visualize(timestamp, entity_path="/timestamp/original")
    datatypes.visualize(later, entity_path="/timestamp/later")

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
