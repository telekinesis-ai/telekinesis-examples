"""Demonstrates the Telekinesis String datatype."""

import time

from loguru import logger
import rerun as rr

from telekinesis import datatypes


def string_example():
    """Demonstrate creation, access, visualization, update, string operations, and serialization."""

    # ======================= Create ============================================
    string = datatypes.String("Hello")
    logger.info(f"Original String: {string}")

    # ======================= Inspect ===========================================
    data = string.data
    logger.info(f"Underlying String data: {data}")

    # ======================= Visualize =========================================
    rr.init("string_example", spawn=True)
    datatypes.visualize(string, entity_path="/String")

    # ======================= Update ============================================
    string.data = "Hello World"
    logger.info(f"Updated String: {string}")

    # ======================= Operations ========================================
    other = datatypes.String("!")
    concatenated = string + other
    repeated = string * 3
    lower = string.lower()
    upper = string.upper()
    stripped = datatypes.String("  Hello World  ").strip()
    contains = "World" in string
    split = datatypes.String("/topic/ns/name").split("/")

    logger.info(f"Concatenation: {string} + {other} = {concatenated}")
    logger.info(f"Repeat: {string} * 3 = {repeated}")
    logger.info(f"Lower: {string}.lower() = {lower}")
    logger.info(f"Upper: {string}.upper() = {upper}")
    logger.info(f"Strip: '  Hello World  '.strip() = {stripped}")
    logger.info(f"Contains: 'World' in {string} = {contains}")
    logger.info(f"Split: {split}")
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
