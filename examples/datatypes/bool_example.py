"""
Example script to demonstrate usage of Bool datatype.
"""

import time

from loguru import logger
import rerun as rr

from telekinesis import datatypes


def bool_example():
    """
    Example function to demonstrate usage of Bool datatype.
        - Create a Bool data
        - Access the underlying native bool
        - Visualize the Bool data using Rerun
        - Perform basic operations on Bool with another Bool
        - Perform basic operations on Bool with native bool(Python)
        - f-string alignment via ``__format__``
        - Serialize to pyarrow and back
    """

    # Construct and access members
    my_bool = datatypes.Bool(True)
    logger.info(f"Original Bool: {my_bool}")

    # Access the underlying native bool
    my_bool_data = my_bool.data
    logger.info(f"Original Bool data: {my_bool_data}")

    logger.info("Visualizing with Rerun...")
    rr.init("bool_example", spawn=True)
    datatypes.visualize(my_bool, entity_path="/Bool")

    # Update the value via the validated setter
    my_bool.data = False
    logger.info(f"Updated Bool: {my_bool}")

    # Operation on Bool with another Bool resulting in Bool object
    my_other_bool = datatypes.Bool(True)

    # AND operation
    my_and_bool = my_bool & my_other_bool
    logger.info(f"Bool AND operation: {my_bool} & {my_other_bool} = {my_and_bool}")

    # OR operation
    my_or_bool = my_bool | my_other_bool
    logger.info(f"Bool OR operation: {my_bool} | {my_other_bool} = {my_or_bool}")

    # XOR operation
    my_xor_bool = my_bool ^ my_other_bool
    logger.info(f"Bool XOR operation: {my_bool} ^ {my_other_bool} = {my_xor_bool}")

    # NOT operation
    my_not_bool = ~my_bool
    logger.info(f"Bool NOT operation: ~{my_bool} = {my_not_bool}")

    # Comparison operation
    my_eq_bool = my_bool == my_other_bool
    logger.info(f"Bool comparison operation: {my_bool} == {my_other_bool} = {my_eq_bool}")

    my_neq_bool = my_bool != my_other_bool
    logger.info(f"Bool comparison operation: {my_bool} != {my_other_bool} = {my_neq_bool}")

    # Operations on Bool with native bool(Python) and resulting in Bool object
    # AND operation
    my_and_native_bool = True & my_bool
    logger.info(f"Bool AND operation with native bool: True & {my_bool} = {my_and_native_bool}")

    # OR operation
    my_or_native_bool = False | my_bool
    logger.info(f"Bool OR operation with native bool: False | {my_bool} = {my_or_native_bool}")

    # XOR operation
    my_xor_native_bool = True ^ my_bool
    logger.info(f"Bool XOR operation with native bool: True ^ {my_bool} = {my_xor_native_bool}")

    # f-strings usage with alignment
    logger.info(f"Bool with f-string format: flag={datatypes.Bool(True):>10}")

    # Serialize to pyarrow and back
    serialization_start_time = time.perf_counter()
    serialized_bool = datatypes.serialize(my_bool)
    serialization_end_time = time.perf_counter()

    deserialization_start_time = time.perf_counter()
    deserialized_bool = datatypes.deserialize(serialized_bool)["param_0"]
    deserialization_end_time = time.perf_counter()

    logger.info(f"Deserialized Bool matches Original: {deserialized_bool == my_bool}")
    logger.info(
        f"Serialization time: {(serialization_end_time - serialization_start_time) * 1000} ms"
    )
    logger.info(
        f"Deserialization time: {(deserialization_end_time - deserialization_start_time) * 1000} ms"
    )


if __name__ == "__main__":
    bool_example()
