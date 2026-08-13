"""
Example script to demonstrate usage of Category datatype.
"""

import time

from loguru import logger
import rerun as rr

from telekinesis import datatypes


def category_example():
    """
    Example function to demonstrate usage of Category datatype.
        - Create a Category
        - Access the underlying id/name/supercategory data
        - Visualize the Category using Rerun
        - Serialize to PyArrow and back
    """
    # Create a Category
    my_category = datatypes.Category(id=1, name="person", supercategory="person")
    logger.info(f"Original Category: {my_category}")

    # Access the underlying id/name/supercategory data
    my_category_id = my_category.id
    my_category_name = my_category.name
    my_category_supercategory = my_category.supercategory
    logger.info(f"Underlying Category id: {my_category_id}")
    logger.info(f"Underlying Category name: {my_category_name}")
    logger.info(f"Underlying Category supercategory: {my_category_supercategory}")

    # Visualize the Category using Rerun
    logger.info("Visualizing with Rerun...")
    rr.init("category_example", spawn=True)
    datatypes.visualize(my_category, entity_path="/Category")

    # Serialize to PyArrow and back
    serialization_start_time = time.perf_counter()
    serialized = datatypes.serialize(my_category)
    serialization_end_time = time.perf_counter()

    deserialization_start_time = time.perf_counter()
    deserialized = datatypes.deserialize(serialized)["param_0"]
    deserialization_end_time = time.perf_counter()

    logger.info(f"Deserialized Category matches Original: {deserialized == my_category}")
    logger.info(
        f"Serialization time: {(serialization_end_time - serialization_start_time) * 1000} ms"
    )
    logger.info(
        f"Deserialization time: {(deserialization_end_time - deserialization_start_time) * 1000} ms"
    )


if __name__ == "__main__":
    category_example()
