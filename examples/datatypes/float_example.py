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
    data = value.data
    logger.info(f"Data: {data}")

    # ======================= Visualize =========================================
    rr.init("float_example", spawn=True)
    datatypes.visualize(value, entity_path="/Float")

    # ======================= Update ============================================
    value.data = 2.71
    logger.info(f"Updated Float: {value}")

    # ======================= Arithmetic ========================================
    other = datatypes.Float(1.5)
    sum_value = value + other
    diff_value = value - other
    prod_value = value * other
    div_value = value / other
    floordiv_value = value // other
    mod_value = value % other

    logger.info(f"{value} + {other} = {sum_value}")
    logger.info(f"{value} - {other} = {diff_value}")
    logger.info(f"{value} * {other} = {prod_value}")
    logger.info(f"{value} / {other} = {div_value}")
    logger.info(f"{value} // {other} = {floordiv_value}")
    logger.info(f"{value} % {other} = {mod_value}")

    # ======================= Convert ===========================================
    native_int = int(value)
    native_float = float(value)
    native_bool = bool(value)

    logger.info(f"int={native_int}, float={native_float}, bool={native_bool}")

    # ======================= Unary =============================================
    negated = -value
    abs_value = abs(value)

    logger.info(f"-{value} = {negated}")
    logger.info(f"abs({value}) = {abs_value}")

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
