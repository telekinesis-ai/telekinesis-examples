"""Demonstrates the Telekinesis String datatype."""

import time

import rerun as rr
from loguru import logger

from telekinesis import datatypes

def string_example():
    """Demonstrate creation, inspection, operations, visualization, and serialization."""

    # ======================= Create ============================================
    value = datatypes.String("Hello")
    logger.info(f"Created String: {value}")

    # ======================= Inspect ===========================================
    logger.info(f"data={value.data}")

    # ======================= Operations ========================================
    value.data = "Hello World"
    logger.info(f"Updated String: {value}")

    other = datatypes.String("!")

    logger.info(f"Concatenation: {value} + {other} = {value + other}")
    logger.info(f"Reflected concatenation: 'Say ' + {value} = {'Say ' + value}")
    logger.info(f"Repeat: {value} * 3 = {value * 3}")
    logger.info(f"Reflected repeat: 3 * {value} = {3 * value}")
    logger.info(f"Lower: {value}.lower() = {value.lower()}")
    logger.info(f"Upper: {value}.upper() = {value.upper()}")

    padded = datatypes.String("  padded  ")
    logger.info(f"Strip: {padded!r}.strip() = {padded.strip()}")

    topic = datatypes.String("/topic/ns/name")
    logger.info(f"Split: {topic}.split('/') = {topic.split('/')}")

    logger.info(f"Length: len({value}) = {len(value)}")
    logger.info(f"Contains: 'World' in {value} = {'World' in value}")
    logger.info(f"Indexing: {value}[0] = {value[0]}")
    logger.info(f"Slicing: {value}[0:5] = {value[0:5]}")
    logger.info(f"Format: '{value:>20}'")

    logger.info(f"str(value)={str(value)}")
    logger.info(f"bool(value)={bool(value)}")

    logger.info(f"EQ: {value} == {other} = {value == other}")
    logger.info(f"LT: {value} < {other} = {value < other}")
    logger.info(f"LE: {value} <= {other} = {value <= other}")
    logger.info(f"GT: {value} > {other} = {value > other}")
    logger.info(f"GE: {value} >= {other} = {value >= other}")

    # ======================= Visualize =========================================
    rr.init("string_example", spawn=True)
    datatypes.visualize(value, entity_path="/string")

    # ======================= Serialize / Deserialize ===========================
    start = time.perf_counter()
    serialized = datatypes.serialize(value)
    serialization_ms = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    deserialized = datatypes.deserialize(serialized)["param_0"]
    deserialization_ms = (time.perf_counter() - start) * 1000

    logger.info(f"Deserialized String: {deserialized}")
    logger.info(f"Round-trip successful: {value == deserialized}")
    logger.info(f"Serialization time: {serialization_ms:.3f} ms")
    logger.info(f"Deserialization time: {deserialization_ms:.3f} ms")


if __name__ == "__main__":
    string_example()
