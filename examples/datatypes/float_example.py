"""Demonstrates the Telekinesis Float datatype."""

import time

import rerun as rr
from loguru import logger

from telekinesis import datatypes

def float_example():
    """Demonstrate creation, inspection, operations, visualization, and serialization."""

    # ======================= Create ============================================
    value = datatypes.Float(3.14)
    logger.info(f"Created Float: {value}")

    # ======================= Inspect ===========================================
    logger.info(f"data={value.data}")

    # ======================= Operations ========================================
    value.data = 2.71
    logger.info(f"Updated Float: {value}")

    other = datatypes.Float(1.5)

    logger.info(f"{value} + {other} = {value + other}")
    logger.info(f"{value} - {other} = {value - other}")
    logger.info(f"{value} * {other} = {value * other}")
    logger.info(f"{value} / {other} = {value / other}")
    logger.info(f"{value} // {other} = {value // other}")
    logger.info(f"{value} % {other} = {value % other}")
    logger.info(f"Reflected add: 1.0 + {value} = {1.0 + value}")
    logger.info(f"Reflected sub: 10.0 - {value} = {10.0 - value}")
    logger.info(f"Reflected mul: 2.0 * {value} = {2.0 * value}")
    logger.info(f"Reflected truediv: 10.0 / {value} = {10.0 / value}")
    logger.info(f"Reflected floordiv: 10.0 // {value} = {10.0 // value}")
    logger.info(f"Reflected mod: 10.0 % {value} = {10.0 % value}")

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

    # ======================= Visualize =========================================
    rr.init("float_example", spawn=True)
    datatypes.visualize(value, entity_path="/float")

    # ======================= Serialize / Deserialize ===========================
    start = time.perf_counter()
    serialized = datatypes.serialize(value)
    serialization_ms = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    deserialized = datatypes.deserialize(serialized)["param_0"]
    deserialization_ms = (time.perf_counter() - start) * 1000

    logger.info(f"Deserialized Float: {deserialized}")
    logger.info(f"Round-trip successful: {deserialized == value}")
    logger.info(f"Serialization time: {serialization_ms:.3f} ms")
    logger.info(f"Deserialization time: {deserialization_ms:.3f} ms")


if __name__ == "__main__":
    float_example()
