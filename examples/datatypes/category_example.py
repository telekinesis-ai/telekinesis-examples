"""Demonstrates the Telekinesis Category datatype."""

import time

from loguru import logger
import rerun as rr

from telekinesis import datatypes

def category_example():
    """Demonstrate creation, access, visualization, and serialization."""

    # ======================= Create ============================================
    category = datatypes.Category(id=1, name="person", supercategory="person")
    logger.info(f"Original Category: {category}")

    # ======================= Inspect ===========================================
    logger.info(
        f"id={category.id}, "
        f"name={category.name}, "
        f"supercategory={category.supercategory}"
    )

    # ======================= Visualize =========================================
    rr.init("category_example", spawn=True)
    datatypes.visualize(category, entity_path="/Category")

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
