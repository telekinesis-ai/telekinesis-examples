"""Demonstrates the Telekinesis Color datatype."""

import time

import numpy as np
import rerun as rr
from loguru import logger

from telekinesis import datatypes

def color_example():
    """Demonstrate creation, inspection, operations, visualization, and serialization."""

    # ======================= Create ============================================
    rgb = [255, 0, 128]
    color = datatypes.Color(rgb)
    logger.info(f"Created Color: {color}")

    hex_color = "#FF00FF80"
    color_from_hex = datatypes.Color.from_hex(hex_color)
    logger.info(f"Color created from hex {hex_color}: {color_from_hex}")

    # ======================= Inspect ===========================================
    logger.info(f"data={color.data}")
    logger.info(f"shape={color.shape}")
    logger.info(f"ndim={color.ndim}")
    logger.info(f"dtype={color.dtype}")
    logger.info(f"size={color.size}")

    # ======================= Operations =========================================
    color.data = [0, 255, 255, 255]
    logger.info(f"Updated Color: {color}")

    color_copy = color.copy()
    logger.info(f"Copied Color: {color_copy}")

    color_numpy = color.to_numpy(copy=True)
    logger.info(f"NumPy Color: {color_numpy}")

    numpy_array = np.asarray(color)
    logger.info(f"NumPy array via __array__: {numpy_array}")

    hex_str = color.to_hex()
    logger.info(f"Color converted to hex: {hex_str}")
    logger.info(f"Hex round-trip successful: {datatypes.Color.from_hex(hex_str) == color}")

    # ======================= Visualize =========================================
    rr.init("color_example", spawn=True)
    datatypes.visualize(color, entity_path="/color/updated")
    datatypes.visualize(color_copy, entity_path="/color/copy")
    datatypes.visualize(color_from_hex, entity_path="/color/from_hex")

    # ======================= Serialize / Deserialize ===========================
    start = time.perf_counter()
    serialized = datatypes.serialize(color)
    serialization_ms = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    deserialized = datatypes.deserialize(serialized)["param_0"]
    deserialization_ms = (time.perf_counter() - start) * 1000

    logger.info(f"Deserialized Color: {deserialized}")
    logger.info(f"Round-trip successful: {color == deserialized}")
    logger.info(f"Serialization time: {serialization_ms:.3f} ms")
    logger.info(f"Deserialization time: {deserialization_ms:.3f} ms")


if __name__ == "__main__":
    color_example()
