"""Demonstrates the Telekinesis Bool datatype."""

import time

import rerun as rr
from loguru import logger

from telekinesis import datatypes

def bool_example():
    """Demonstrate creation, inspection, operations, visualization, and serialization."""

    # ======================= Create ============================================
    value = datatypes.Bool(True)
    logger.info(f"Created Bool: {value}")

    # ======================= Inspect ===========================================
    logger.info(f"data={value.data}")

    # ======================= Operations ========================================
    value.data = False
    logger.info(f"Updated Bool: {value}")

    other = datatypes.Bool(True)

    logger.info(f"AND: {value} & {other} = {value & other}")
    logger.info(f"OR: {value} | {other} = {value | other}")
    logger.info(f"XOR: {value} ^ {other} = {value ^ other}")
    logger.info(f"NOT: ~{value} = {~value}")
    logger.info(f"Reflected AND: True & {value} = {True & value}")
    logger.info(f"Reflected OR: False | {value} = {False | value}")
    logger.info(f"Reflected XOR: True ^ {value} = {True ^ value}")
    logger.info(f"EQ: {value} == {other} = {value == other}")
    logger.info(f"int(value)={int(value)}")
    logger.info(f"bool(value)={bool(value)}")

    # ======================= Visualize =========================================
    rr.init("bool_example", spawn=True)
    datatypes.visualize(value, entity_path="/bool")

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
