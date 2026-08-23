"""Demonstrates the Telekinesis Color datatype."""

import time

import rerun as rr
from loguru import logger

from telekinesis import datatypes

def rgba32_example():
    """Demonstrate creation, inspection, visualization, hex conversion, and serialization."""

    # ======================= Create ============================================
    rgb = [255, 0, 128]
    color = datatypes.Color(rgb)
    logger.info(f"Original Color: {color}")

    # ======================= Inspect ===========================================
    logger.info(
        f"shape={color.shape}, "
        f"size={color.size}, "
        f"ndim={color.ndim}, "
        f"dtype={color.dtype}"
    )
    logger.info(f"Color data: {color.data}")
    logger.info(f"NumPy array: {color.to_numpy()}")
    logger.info(f"Copied Color: {color.copy()}")

    # ======================= Visualize =========================================
    rr.init("rgba32_example", spawn=True)
    datatypes.visualize(color, entity_path="/Color")

    # ======================= Update ============================================
    color.data = [0, 255, 255, 255]
    logger.info(f"Updated Color: {color}")
    datatypes.visualize(color, entity_path="/Color/updated")

    # ======================= From Hex ==========================================
    hex_color = "#FF00FF80"
    color_from_hex = datatypes.Color.from_hex(hex_color)

    logger.info(f"Color from hex {hex_color}: {color_from_hex}")
    logger.info(f"Hex round-trip successful: {color_from_hex.to_hex() == hex_color}")
    datatypes.visualize(color_from_hex, entity_path="/Color/from_hex")

    # ======================= Serialize / Deserialize ===========================
    start = time.perf_counter()
    serialized = datatypes.serialize(color)
    serialization_ms = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    deserialized = datatypes.deserialize(serialized)["param_0"]
    deserialization_ms = (time.perf_counter() - start) * 1000

    logger.info(f"Deserialized Color: {deserialized}")
    logger.info(f"Round-trip successful: {deserialized == color}")
    logger.info(f"Serialization time: {serialization_ms:.3f} ms")
    logger.info(f"Deserialization time: {deserialization_ms:.3f} ms")


if __name__ == "__main__":
    rgba32_example()
