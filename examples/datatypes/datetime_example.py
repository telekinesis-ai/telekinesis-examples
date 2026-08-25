"""Demonstrates the Telekinesis DateTime datatype."""

import time
from datetime import datetime, timedelta, timezone

import rerun as rr
from loguru import logger

from telekinesis import datatypes

def datetime_example():
    """Demonstrate creation, inspection, operations, visualization, and serialization."""

    # ======================= Create ============================================
    created_at = datatypes.DateTime(datetime.now(timezone.utc))
    logger.info(f"Created DateTime: {created_at}")

    # ======================= Inspect ===========================================
    logger.info(f"data={created_at.data}")

    # ======================= Operations ========================================
    created_at.data = created_at.data + timedelta(minutes=5)
    logger.info(f"Updated DateTime: {created_at}")

    # Any timezone-aware datetime is accepted; it is normalized to UTC on storage.
    pst = datatypes.DateTime(datetime.now(timezone(timedelta(hours=-8))))
    logger.info(f"DateTime from PST input, normalized to UTC: {pst}")

    earlier = datatypes.DateTime(created_at.data - timedelta(minutes=1))

    logger.info(f"EQ: {created_at} == {created_at} = {created_at == created_at}")
    logger.info(f"LT: {earlier} < {created_at} = {earlier < created_at}")
    logger.info(f"GT (via reflected __lt__): {created_at} > {earlier} = {created_at > earlier}")

    # ======================= Visualize =========================================
    rr.init("datetime_example", spawn=True)
    datatypes.visualize(created_at, entity_path="/datetime/updated")
    datatypes.visualize(pst, entity_path="/datetime/pst")

    # ======================= Serialize / Deserialize ===========================
    start = time.perf_counter()
    serialized = datatypes.serialize(created_at)
    serialization_ms = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    deserialized = datatypes.deserialize(serialized)["param_0"]
    deserialization_ms = (time.perf_counter() - start) * 1000

    logger.info(f"Deserialized DateTime: {deserialized}")
    logger.info(f"Round-trip successful: {created_at == deserialized}")
    logger.info(f"Serialization time: {serialization_ms:.3f} ms")
    logger.info(f"Deserialization time: {deserialization_ms:.3f} ms")


if __name__ == "__main__":
    datetime_example()
