"""
Example script to demonstrate usage of Int datatype.
"""

import time

from loguru import logger
import rerun as rr

from telekinesis import datatypes


def int_example():
    """
    Example function to demonstrate usage of Int datatype.
        - Create an Int data
        - Access the underlying native int through ``.data``
        - Visualize the Int data using Rerun
        - Update the value via the validated setter
        - Arithmetic operations: `+`, `-`, `*`, `/`, `%`
        - Convert to native `int`/`float`/`bool` and use as a list index
        - Unary operations: negation and abs
        - f-string alignment and hex formatting
        - Serialize to PyArrow and back
    """

    # Create an Int data
    my_int = datatypes.Int(42)
    logger.info(f"Original Int: {my_int}")

    # Access the underlying data
    my_int_data = my_int.data
    logger.info(f"Original Int data: {my_int_data}")

    logger.info("Visualizing with Rerun...")
    rr.init("int_example", spawn=True)
    datatypes.visualize(my_int, entity_path="/Int")

    # Update the value of Int
    my_int.data = 100
    logger.info(f"Updated Int: {my_int}")

    # Operation on Int with another Int resulting in a new Int/Float object
    my_other_int = datatypes.Int(58)

    # Arithmetic operations on Int
    # Addition operation
    my_sum_int = my_int + my_other_int
    logger.info(f"Int Addition operation: {my_int} + {my_other_int} = {my_sum_int}")

    # Subtraction operation
    my_diff_int = my_int - my_other_int
    logger.info(f"Int Subtraction operation: {my_int} - {my_other_int} = {my_diff_int}")

    # Multiplication operation
    my_prod_int = my_int * my_other_int
    logger.info(f"Int Multiplication operation: {my_int} * {my_other_int} = {my_prod_int}")

    # Division operation resulting in a new Float object
    my_div_int = my_int / my_other_int
    logger.info(f"Int Division operation: {my_int} / {my_other_int} = {my_div_int}")

    # Modulus operation
    my_mod_int = my_int % my_other_int
    logger.info(f"Int Modulus operation: {my_int} % {my_other_int} = {my_mod_int}")

    # Convert to native int
    native_int = int(my_int)
    logger.info(f"Converted to native int: {native_int}")

    # Convert to native float
    native_float = float(my_int)
    logger.info(f"Converted to native float: {native_float}")

    # Convert to native bool
    native_bool = bool(my_int)
    logger.info(f"Converted to native bool: {native_bool}")

    # Use as a list index
    my_list = [1, 2, 3, 4, 5]
    index = my_int % len(my_list)  # Use the underlying int value for indexing
    logger.info(f"Accessing list with Int as index: my_list[{index}] = {my_list[index]}")

    # Unary operations on Int
    negated_int = -my_int
    logger.info(f"Negated Int: -{my_int} = {negated_int}")

    # Absolute value operation
    abs_int = abs(my_int)
    logger.info(f"Absolute Int: abs({my_int}) = {abs_int}")

    # Compare Int with another Int
    comparison_result = my_int > my_other_int
    logger.info(f"Comparison operation: {my_int} > {my_other_int} = {comparison_result}")

    # fstring usage
    logger.info(f"f-string format: id={datatypes.Int(42):05d}")
    logger.info(f"f-string format: hex={datatypes.Int(255):x}")

    # Serialize to pyarrow and back
    serialization_start_time = time.perf_counter()
    serialized = datatypes.serialize(my_int)
    serialization_end_time = time.perf_counter()

    deserialization_start_time = time.perf_counter()
    deserialized = datatypes.deserialize(serialized)["param_0"]
    deserialization_end_time = time.perf_counter()

    logger.info(f"Deserialized Int matches Original: {deserialized == my_int}")

    logger.info(
        f"Serialization time: {(serialization_end_time - serialization_start_time) * 1000} ms"
    )
    logger.info(
        f"Deserialization time: {(deserialization_end_time - deserialization_start_time) * 1000} ms"
    )


if __name__ == "__main__":
    int_example()
