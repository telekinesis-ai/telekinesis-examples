"""Demonstrates the Telekinesis Timestamp datatype."""

import time

from loguru import logger
import rerun as rr

from telekinesis import datatypes


def timestamp_example():
    """Demonstrate creation, access, visualization, comparison, and serialization."""

    # ======================= Create ============================================
    timestamp = datatypes.Timestamp(sec=42, nanosec=250_000_000)
    logger.info(f"Original Timestamp: {timestamp}")

    # ======================= Inspect ===========================================
    logger.info(f"sec={timestamp.sec}, nanosec={timestamp.nanosec}")

    # ======================= Visualize =========================================
    rr.init("timestamp_example", spawn=True)
    datatypes.visualize(timestamp, entity_path="/Timestamp")

    # ======================= New Instance ======================================
    # Timestamp is immutable; build a new instance rather than mutating in place.
    later = datatypes.Timestamp(sec=43, nanosec=0)
    logger.info(f"Later Timestamp: {later}")
    datatypes.visualize(later, entity_path="/Timestamp/later")

    # ======================= Compare ===========================================
    diff_sec = (later.sec + later.nanosec / 1e9) - (timestamp.sec + timestamp.nanosec / 1e9)
    logger.info(f"Difference between timestamps: {diff_sec:.3f} s")
    logger.info(f"EQ: {timestamp} == {timestamp} = {timestamp == datatypes.Timestamp(sec=42, nanosec=250_000_000)}")

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
