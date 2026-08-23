"""Demonstrates the Telekinesis Category datatype."""

import time

import rerun as rr
from loguru import logger

from telekinesis import datatypes

def category_example():
    """Demonstrate creation, inspection, operations, visualization, and serialization."""

    # ======================= Create ============================================
    category = datatypes.Category(id=1, name="person", supercategory="person")
    logger.info(f"Created Category: {category}")

    # ======================= Inspect ===========================================
    logger.info(f"id={category.id}")
    logger.info(f"name={category.name}")
    logger.info(f"supercategory={category.supercategory}")

    # ======================= Operations =========================================
    other_category = datatypes.Category(id=2, name="bicycle", supercategory="vehicle")
    logger.info(f"category == category: {category == category}")
    logger.info(f"category == other_category: {category == other_category}")

    # ======================= Visualize =========================================
    rr.init("category_example", spawn=True)
    datatypes.visualize(category, entity_path="/category")

    # ======================= Serialize / Deserialize ===========================
    start = time.perf_counter()
    serialized = datatypes.serialize(category)
    serialization_ms = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    deserialized = datatypes.deserialize(serialized)["param_0"]
    deserialization_ms = (time.perf_counter() - start) * 1000

    logger.info(f"Deserialized Category: {deserialized}")
    logger.info(f"Round-trip successful: {category == deserialized}")
    logger.info(f"Serialization time: {serialization_ms:.3f} ms")
    logger.info(f"Deserialization time: {deserialization_ms:.3f} ms")


if __name__ == "__main__":
    category_example()
