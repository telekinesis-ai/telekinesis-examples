"""
Example script to demonstrate usage of String datatype.
"""

import time

from loguru import logger
import rerun as rr

from telekinesis import datatypes


def string_example():
    """
    Example function to demonstrate usage of String datatype.
        - Create a String data
    """

    # Construct and access members
    my_string = datatypes.String("Hello")
    logger.info(f"Original String: {my_string}")

    # Access the underlying data
    my_string_data = my_string.data
    logger.info(f"Original String data: {my_string_data}")

    logger.info("Visualizing with Rerun...")
    rr.init("string_example", spawn=True)
    datatypes.visualize(my_string, entity_path="/String")

    # Update the value
    my_string.data = "Hello World"
    logger.info(f"Updated String: {my_string}")

    # Operations on String with another String resulting in a new String object
    my_other_string = datatypes.String("!")
    my_concat_string = my_string + my_other_string
    logger.info(
        f"String Concatenation operation: {my_string} + {my_other_string} = {my_concat_string}"
    )

    # Repeat through operator *
    my_repeat_string = my_string * 3
    logger.info(f"String Repeat operation: {my_string} * 3 = {my_repeat_string}")

    # Lowercase
    my_lower_string = my_string.lower()
    logger.info(f"String Lowercase operation: {my_string}.lower() = {my_lower_string}")

    # Uppercase
    my_upper_string = my_string.upper()
    logger.info(f"String Uppercase operation: {my_string}.upper() = {my_upper_string}")

    # Strip
    my_strip_string = datatypes.String("  Hello World  ").strip()
    logger.info(f'String Strip operation: String("  Hello World  ").strip() = {my_strip_string}')

    # Contains operation (in)
    my_contains_bool = "World" in my_string
    logger.info(f'String Contains operation: "World" in {my_string} = {my_contains_bool}')

    # Fstring usage
    logger.info(f"Split on /topic/ns/name: {datatypes.String('/topic/ns/name').split('/')}")
    logger.info(f"Format with f-string format: {my_strip_string:>10}")

    # Serialize to pyarrow and back
    serialization_start_time = time.perf_counter()
    serialized = datatypes.serialize(my_string)
    serialization_end_time = time.perf_counter()

    deserialization_start_time = time.perf_counter()
    deserialized = datatypes.deserialize(serialized)["param_0"]
    deserialization_end_time = time.perf_counter()

    logger.info(f"Deserialized String matches Original: {deserialized == my_string}")
    logger.info(
        f"Serialization time: {(serialization_end_time - serialization_start_time) * 1000} ms"
    )
    logger.info(
        f"Deserialization time: {(deserialization_end_time - deserialization_start_time) * 1000} ms"
    )


if __name__ == "__main__":
    string_example()
