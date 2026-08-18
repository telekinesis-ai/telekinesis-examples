"""Demonstrates the Telekinesis Bool datatype."""

import time

from loguru import logger
import rerun as rr

from telekinesis import datatypes

def bool_example():
    """Demonstrate creation, access, boolean operations, formatting, and serialization."""

    # ======================= Create ============================================
    value = datatypes.Bool(True)
    data = value.data

    logger.info(f"Original Bool: {value}")
    logger.info(f"Bool data: {data}")

    # ======================= Visualize =========================================
    rr.init("bool_example", spawn=True)
    datatypes.visualize(value, entity_path="/Bool")

    # ======================= Update ============================================
    value.data = False
    logger.info(f"Updated Bool: {value}")

    # ======================= Operations ========================================
    other = datatypes.Bool(True)
    and_result = value & other
    or_result = value | other
    xor_result = value ^ other
    not_result = ~value
    eq_result = value == other
    neq_result = value != other

    logger.info(f"AND: {value} & {other} = {and_result}")
    logger.info(f"OR: {value} | {other} = {or_result}")
    logger.info(f"XOR: {value} ^ {other} = {xor_result}")
    logger.info(f"NOT: ~{value} = {not_result}")
    logger.info(f"EQ: {value} == {other} = {eq_result}")
    logger.info(f"NEQ: {value} != {other} = {neq_result}")

    # ======================= Native Operations =================================
    and_native = True & value
    or_native = False | value
    xor_native = True ^ value

    logger.info(f"AND with native bool: True & {value} = {and_native}")
    logger.info(f"OR with native bool: False | {value} = {or_native}")
    logger.info(f"XOR with native bool: True ^ {value} = {xor_native}")

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
