"""Demonstrates the Telekinesis String datatype."""

import time

import rerun as rr
from loguru import logger

from telekinesis import datatypes


def string_example():
    """Demonstrate creation, access, visualization, update, string operations, and serialization."""

    # ======================= Create ============================================
    string = datatypes.String("Hello")
    logger.info(f"Original String: {string}")

    # ======================= Inspect ===========================================
    logger.info(f"String data: {string.data}")

    # ======================= Visualize =========================================
    rr.init("string_example", spawn=True)
    datatypes.visualize(string, entity_path="/String")

    # ======================= Update ============================================
    string.data = "Hello World"
    logger.info(f"Updated String: {string}")

    # ======================= Operations ========================================
    other = datatypes.String("!")
    stripped = datatypes.String("  Hello World  ").strip()

    logger.info(f"Concatenation: {string} + {other} = {string + other}")
    logger.info(f"Repeat: {string} * 3 = {string * 3}")
    logger.info(f"Lower: {string}.lower() = {string.lower()}")
    logger.info(f"Upper: {string}.upper() = {string.upper()}")
    logger.info(f"Strip: '  Hello World  '.strip() = {stripped}")
    logger.info(f"Contains: 'World' in {string} = {'World' in string}")
    logger.info(f"Split: {datatypes.String('/topic/ns/name').split('/')}")
    logger.info(f"Format with f-string format: {stripped:>10}")

    # ======================= Serialize / Deserialize ===========================
    start = time.perf_counter()
    serialized = datatypes.serialize(string)
    serialization_ms = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    deserialized = datatypes.deserialize(serialized)["param_0"]
    deserialization_ms = (time.perf_counter() - start) * 1000

    logger.info(f"Deserialized String: {deserialized}")
    logger.info(f"Round-trip successful: {string == deserialized}")
    logger.info(f"Serialization time: {serialization_ms:.3f} ms")
    logger.info(f"Deserialization time: {deserialization_ms:.3f} ms")


if __name__ == "__main__":
    string_example()
