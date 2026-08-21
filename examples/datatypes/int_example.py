"""Demonstrates the Telekinesis Int datatype."""

import time

from loguru import logger
import rerun as rr

from telekinesis import datatypes

def int_example():
    """Demonstrate creation, access, arithmetic, conversion, and serialization."""

    # ======================= Create ============================================
    value = datatypes.Int(42)

    logger.info(f"Created Int: {value}")

    # ======================= Inspect ===========================================
    logger.info(f"Int data: {value.data}")

    # ======================= Visualize =========================================
    rr.init("int_example", spawn=True)
    datatypes.visualize(value, entity_path="/Int")

    # ======================= Update ============================================
    value.data = 100

    logger.info(f"Updated Int: {value}")

    # ======================= Arithmetic ========================================
    other = datatypes.Int(58)

    logger.info(f"{value} + {other} = {value + other}")
    logger.info(f"{value} - {other} = {value - other}")
    logger.info(f"{value} * {other} = {value * other}")
    logger.info(f"{value} / {other} = {value / other}")
    logger.info(f"{value} % {other} = {value % other}")

    # ======================= Convert ===========================================
    logger.info(f"int={int(value)}, float={float(value)}, bool={bool(value)}")

    # ======================= Index =============================================
    items = [1, 2, 3, 4, 5]
    index = value % len(items)

    logger.info(f"items[{index}] = {items[index]}")

    # ======================= Unary =============================================
    logger.info(f"negated={-value}, absolute={abs(value)}")

    # ======================= Compare ===========================================
    logger.info(f"{value} > {other} = {value > other}")

    # ======================= Format ============================================
    logger.info(f"id={datatypes.Int(42):05d}")
    logger.info(f"hex={datatypes.Int(255):x}")

    # ======================= Serialize / Deserialize ===========================
    start = time.perf_counter()
    serialized = datatypes.serialize(value)
    serialization_ms = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    deserialized = datatypes.deserialize(serialized)["param_0"]
    deserialization_ms = (time.perf_counter() - start) * 1000

    logger.info(f"Deserialized Int: {deserialized}")
    logger.info(f"Round-trip successful: {deserialized == value}")
    logger.info(f"Serialization time: {serialization_ms:.3f} ms")
    logger.info(f"Deserialization time: {deserialization_ms:.3f} ms")


if __name__ == "__main__":
    int_example()
