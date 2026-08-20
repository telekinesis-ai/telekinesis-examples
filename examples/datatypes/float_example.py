"""Demonstrates the Telekinesis Float datatype."""

import time

from loguru import logger
import rerun as rr

from telekinesis import datatypes

def float_example():
    """Demonstrate creation, inspection, visualization, update, arithmetic, native conversion, unary ops, formatting, and serialization."""

    # ======================= Create ============================================
    value = datatypes.Float(3.14)
    logger.info(f"Original Float: {value}")

    # ======================= Inspect ===========================================
    logger.info(f"Data: {value.data}")

    # ======================= Visualize =========================================
    rr.init("float_example", spawn=True)
    datatypes.visualize(value, entity_path="/Float")

    # ======================= Update ============================================
    value.data = 2.71
    logger.info(f"Updated Float: {value}")

    # ======================= Arithmetic ========================================
    other = datatypes.Float(1.5)

    logger.info(f"{value} + {other} = {value + other}")
    logger.info(f"{value} - {other} = {value - other}")
    logger.info(f"{value} * {other} = {value * other}")
    logger.info(f"{value} / {other} = {value / other}")
    logger.info(f"{value} // {other} = {value // other}")
    logger.info(f"{value} % {other} = {value % other}")

    # ======================= Convert ===========================================
    logger.info(f"int={int(value)}, float={float(value)}, bool={bool(value)}")

    # ======================= Unary =============================================
    logger.info(f"-{value} = {-value}")
    logger.info(f"abs({value}) = {abs(value)}")

    # ======================= Format ============================================
    logger.info(f"f-string format: pose={datatypes.Float(3.14159):.3f}")
    logger.info(f"f-string format: percent={datatypes.Float(0.5):.1%}")

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
