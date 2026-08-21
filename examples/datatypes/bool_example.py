"""Demonstrates the Telekinesis Bool datatype."""

import time

from loguru import logger
import rerun as rr

from telekinesis import datatypes

def bool_example():
    """Demonstrate creation, access, boolean operations, formatting, and serialization."""

    # ======================= Create ============================================
    value = datatypes.Bool(True)

    logger.info(f"Original Bool: {value}")

    # ======================= Inspect ===========================================
    logger.info(f"Bool data: {value.data}")

    # ======================= Visualize =========================================
    rr.init("bool_example", spawn=True)
    datatypes.visualize(value, entity_path="/Bool")

    # ======================= Update ============================================
    value.data = False
    logger.info(f"Updated Bool: {value}")

    # ======================= Operations ========================================
    other = datatypes.Bool(True)

    logger.info(f"AND: {value} & {other} = {value & other}")
    logger.info(f"OR: {value} | {other} = {value | other}")
    logger.info(f"XOR: {value} ^ {other} = {value ^ other}")
    logger.info(f"NOT: ~{value} = {~value}")
    logger.info(f"EQ: {value} == {other} = {value == other}")
    logger.info(f"NEQ: {value} != {other} = {value != other}")

    # ======================= Native Operations =================================
    logger.info(f"AND with native bool: True & {value} = {True & value}")
    logger.info(f"OR with native bool: False | {value} = {False | value}")
    logger.info(f"XOR with native bool: True ^ {value} = {True ^ value}")

    # ======================= Format ============================================
    logger.info(f"f-string format: flag={datatypes.Bool(True):>10}")

    # ======================= Serialize / Deserialize ===========================
    start = time.perf_counter()
    serialized = datatypes.serialize(value)
    serialization_ms = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    deserialized = datatypes.deserialize(serialized)["param_0"]
    deserialization_ms = (time.perf_counter() - start) * 1000

    logger.info(f"Deserialized Bool: {deserialized}")
    logger.info(f"Round-trip successful: {value == deserialized}")
    logger.info(f"Serialization time: {serialization_ms:.3f} ms")
    logger.info(f"Deserialization time: {deserialization_ms:.3f} ms")


if __name__ == "__main__":
    bool_example()
