"""Demonstrates the Telekinesis Int datatype."""

import time

import rerun as rr
from loguru import logger

from telekinesis import datatypes

def int_example():
    """Demonstrate creation, inspection, operations, visualization, and serialization."""

    # ======================= Create ============================================
    value = datatypes.Int(42)
    logger.info(f"Created Int: {value}")

    # ======================= Inspect ===========================================
    logger.info(f"data={value.data}")

    # ======================= Operations ========================================
    value.data = 100
    logger.info(f"Updated Int: {value}")

    other = datatypes.Int(58)

    logger.info(f"{value} + {other} = {value + other}")
    logger.info(f"{value} - {other} = {value - other}")
    logger.info(f"{value} * {other} = {value * other}")
    logger.info(f"{value} / {other} = {value / other}")
    logger.info(f"{value} // {other} = {value // other}")
    logger.info(f"{value} % {other} = {value % other}")
    logger.info(f"Reflected add: 10 + {value} = {10 + value}")
    logger.info(f"Reflected sub: 200 - {value} = {200 - value}")
    logger.info(f"Reflected mul: 2 * {value} = {2 * value}")
    logger.info(f"Reflected truediv: 1000 / {value} = {1000 / value}")
    logger.info(f"Reflected floordiv: 1000 // {value} = {1000 // value}")
    logger.info(f"Reflected mod: 1000 % {value} = {1000 % value}")

    logger.info(f"negated={-value}")
    logger.info(f"positive={+value}")
    logger.info(f"absolute={abs(value)}")

    logger.info(f"EQ: {value} == {other} = {value == other}")
    logger.info(f"LT: {value} < {other} = {value < other}")
    logger.info(f"LE: {value} <= {other} = {value <= other}")
    logger.info(f"GT: {value} > {other} = {value > other}")
    logger.info(f"GE: {value} >= {other} = {value >= other}")

    logger.info(f"int(value)={int(value)}")
    logger.info(f"float(value)={float(value)}")
    logger.info(f"bool(value)={bool(value)}")

    items = [1, 2, 3, 4, 5]
    index = value % len(items)
    logger.info(f"Used as index via __index__: items[{index}] = {items[index]}")

    # ======================= Visualize =========================================
    rr.init("int_example", spawn=True)
    datatypes.visualize(value, entity_path="/int")

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
