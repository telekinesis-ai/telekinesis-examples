"""Demonstrates the Telekinesis Categories datatype."""

import time

import numpy as np
import rerun as rr
from loguru import logger

from telekinesis import datatypes

def categories_example():
    """Demonstrate creation, access, indexing, rebuilding the immutable table, NumPy-based ranking, and serialization."""

    # ======================= Create ============================================
    categories = datatypes.Categories(
        ids=np.array([1, 2, 3], dtype=np.int32),
        names=np.array(["person", "bicycle", "car"]),
        supercategories=np.array(["person", "vehicle", "vehicle"]),
    )
    logger.info(f"Original Categories: {categories}")

    # ======================= Inspect ===========================================
    logger.info(f"length={len(categories)}")
    logger.info(
        f"ids={categories.ids}, "
        f"names={categories.names}, "
        f"supercategories={categories.supercategories}"
    )

    # ======================= Index =============================================
    category_id = 2
    logger.info(f"Indexed category for id {category_id}: {categories[category_id]}")

    # ======================= Visualize =========================================
    rr.init("categories_example", spawn=True)
    datatypes.visualize(categories, entity_path="/Categories")

    # ======================= Rebuild ===========================================
    updated = datatypes.Categories(
        ids=np.append(categories.ids, 4),
        names=np.append(categories.names, "traffic light"),
        supercategories=np.append(categories.supercategories, "outdoor"),
    )
    logger.info(f"Updated Categories: {updated}")
    datatypes.visualize(updated, entity_path="/Categories/updated")

    # ======================= NumPy Interop =====================================
    order = np.argsort(-updated.ids)
    logger.info(
        f"Categories ranked by id (descending): {updated.names[order]} "
        f"(order: {order.tolist()})"
    )

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
