"""Demonstrates the Telekinesis Categories datatype."""

import time

import numpy as np
import rerun as rr
from loguru import logger

from telekinesis import datatypes

def categories_example():
    """Demonstrate creation, inspection, operations, visualization, and serialization."""

    # ======================= Create ============================================
    categories = datatypes.Categories(
        ids=np.array([1, 2, 3], dtype=np.int32),
        names=np.array(["person", "bicycle", "car"]),
        supercategories=np.array(["person", "vehicle", "vehicle"]),
    )
    logger.info(f"Created Categories: {categories}")

    # ======================= Inspect ===========================================
    logger.info(f"ids={categories.ids}")
    logger.info(f"names={categories.names}")
    logger.info(f"supercategories={categories.supercategories}")

    # ======================= Operations =========================================
    logger.info(f"length={len(categories)}")

    # int indexing returns the Category at that position (not an id lookup)
    index = 1
    logger.info(f"categories[{index}] = {categories[index]}")

    sliced = categories[0:2]
    logger.info(f"categories[0:2] = {sliced}")

    mask = categories.ids >= 2
    masked = categories[mask]
    logger.info(f"categories[ids >= 2] = {masked}")

    # Categories is immutable; build a new instance to add or change entries.
    updated = datatypes.Categories(
        ids=np.append(categories.ids, 4),
        names=np.append(categories.names, "traffic light"),
        supercategories=np.append(categories.supercategories, "outdoor"),
    )
    logger.info(f"Updated Categories: {updated}")

    order = np.argsort(-updated.ids)
    logger.info(f"Categories ranked by id (descending): {updated.names[order]}")

    # ======================= Visualize =========================================
    rr.init("categories_example", spawn=True)
    datatypes.visualize(categories, entity_path="/categories/original")
    datatypes.visualize(updated, entity_path="/categories/updated")

    # ======================= Serialize / Deserialize ===========================
    start = time.perf_counter()
    serialized = datatypes.serialize(updated)
    serialization_ms = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    deserialized = datatypes.deserialize(serialized)["param_0"]
    deserialization_ms = (time.perf_counter() - start) * 1000

    logger.info(f"Deserialized Categories: {deserialized}")
    logger.info(f"Round-trip successful: {updated == deserialized}")
    logger.info(f"Serialization time: {serialization_ms:.3f} ms")
    logger.info(f"Deserialization time: {deserialization_ms:.3f} ms")


if __name__ == "__main__":
    categories_example()
