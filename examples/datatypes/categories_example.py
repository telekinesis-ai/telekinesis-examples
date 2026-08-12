"""
Example script to demonstrate usage of Categories datatype.
"""

import time

import numpy as np
from loguru import logger
import rerun as rr

from telekinesis import datatypes


def categories_example():
    """
    Example function to demonstrate usage of Categories datatype.
        - Create a Categories table from ids/names/supercategories
        - Access the underlying fields (ids, names, supercategories)
        - Inspect the table with ``len()``
        - Build a new table, since Categories is immutable after construction
        - Use with numpy to filter/rank categories
        - Serialize to PyArrow and back
    """

    # Create a Categories table
    my_categories = datatypes.Categories(
        ids=np.array([1, 2, 3], dtype=np.int32),
        names=np.array(["person", "bicycle", "car"]),
        supercategories=np.array(["person", "vehicle", "vehicle"]),
    )
    logger.info(f"Original Categories: {my_categories}")

    # Access the underlying data
    my_categories_len = len(my_categories)
    my_categories_ids = my_categories.ids
    my_categories_names = my_categories.names
    my_categories_supercategories = my_categories.supercategories

    logger.info(f"Underlying Categories len: {my_categories_len}")
    logger.info(f"Underlying Categories ids: {my_categories_ids}")
    logger.info(f"Underlying Categories names: {my_categories_names}")
    logger.info(f"Underlying Categories supercategories: {my_categories_supercategories}")

    # Get a single category by category id
    category_id_to_lookup = 2
    my_indexed_category = my_categories[category_id_to_lookup]
    logger.info(f"Indexed category for id {category_id_to_lookup}: {my_indexed_category}")

    logger.info("Visualizing with Rerun...")
    rr.init("categories_example", spawn=True)
    datatypes.visualize(my_categories, entity_path="/Categories")

    # Categories is immutable after construction (its fields are interdependent
    # Build a new Categories table with an additional category
    updated_categories = datatypes.Categories(
        ids=np.append(my_categories.ids, 4),
        names=np.append(my_categories.names, "traffic light"),
        supercategories=np.append(my_categories.supercategories, "outdoor"),
    )
    logger.info(f"Updated Categories: {updated_categories}")
    datatypes.visualize(updated_categories, entity_path="/Categories/updated")

    # Operate with numpy: rank categories by id, descending
    order = np.argsort(-updated_categories.ids)
    logger.info(
        f"Categories ranked by id (descending): {updated_categories.names[order]} "
        f"(order: {order.tolist()})"
    )

    # Serialize to PyArrow and back
    serialization_start_time = time.perf_counter()
    serialized = datatypes.serialize(updated_categories)
    serialization_end_time = time.perf_counter()

    deserialization_start_time = time.perf_counter()
    deserialized = datatypes.deserialize(serialized)["param_0"]
    deserialization_end_time = time.perf_counter()
    logger.info(f"Deserialized Categories: {deserialized}")
    logger.info(
        f"Deserialized Categories and original Categories are equal: {deserialized == updated_categories}"
    )

    logger.info(
        f"Serialization time: {(serialization_end_time - serialization_start_time) * 1000} ms"
    )
    logger.info(
        f"Deserialization time: {(deserialization_end_time - deserialization_start_time) * 1000} ms"
    )


if __name__ == "__main__":
    categories_example()
