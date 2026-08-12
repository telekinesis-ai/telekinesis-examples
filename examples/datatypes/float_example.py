"""
Example script to demonstrate usage of Float datatype.
"""

import time

from loguru import logger
import rerun as rr

from telekinesis import datatypes


def float_example():
    """
    Example function to demonstrate usage of Float datatype.
        - Create a Float data
        - Access the underlying native float through ``.data``
        - Visualize the Float data using Rerun
        - Update the value via the validated setter
        - Arithmetic operations: `+`, `-`, `*`, `/`, `//`, `%`
        - Convert to native `int`/`float`/`bool`
        - Unary operations: negation and ``abs``
        - f-string formatting via ``__format__``
        - Serialize to PyArrow and back
    """

    # Create a Float data
    my_float = datatypes.Float(3.14)
    logger.info(f"Original Float: {my_float}")

    # Access the underlying data
    my_float_data = my_float.data
    logger.info(f"Original Float data: {my_float_data}")

    logger.info("Visualizing with Rerun...")
    rr.init("float_example", spawn=True)
    datatypes.visualize(my_float, entity_path="/Float")

    # Update the value via the validated setter
    my_float.data = 2.71
    logger.info(f"Updated Float: {my_float}")

    # Operation on Float with another Float resulting in a new Float object
    my_other_float = datatypes.Float(1.5)

    # Arithmetic operations on Float
    # Addition operation
    my_sum_float = my_float + my_other_float
    logger.info(f"Float Addition operation: {my_float} + {my_other_float} = {my_sum_float}")

    # Subtraction operation
    my_diff_float = my_float - my_other_float
    logger.info(f"Float Subtraction operation: {my_float} - {my_other_float} = {my_diff_float}")

    # Multiplication operation
    my_prod_float = my_float * my_other_float
    logger.info(f"Float Multiplication operation: {my_float} * {my_other_float} = {my_prod_float}")

    # Division operation
    my_div_float = my_float / my_other_float
    logger.info(f"Float Division operation: {my_float} / {my_other_float} = {my_div_float}")

    # Floor division operation
    my_floordiv_float = my_float // my_other_float
    logger.info(
        f"Float Floor division operation: {my_float} // {my_other_float} = {my_floordiv_float}"
    )

    # Modulus operation
    my_mod_float = my_float % my_other_float
    logger.info(f"Float Modulus operation: {my_float} % {my_other_float} = {my_mod_float}")

    # Convert to native int
    native_int = int(my_float)
    logger.info(f"Converted to native int: {native_int}")

    # Convert to native float
    native_float = float(my_float)
    logger.info(f"Converted to native float: {native_float}")

    # Convert to native bool
    native_bool = bool(my_float)
    logger.info(f"Converted to native bool: {native_bool}")

    # Unary operations on Float
    negated_float = -my_float
    logger.info(f"Negated Float: -{my_float} = {negated_float}")

    # Absolute value operation
    abs_float = abs(my_float)
    logger.info(f"Absolute Float: abs({my_float}) = {abs_float}")

    # fstring usage
    logger.info(f"f-string format: pose={datatypes.Float(3.14159):.3f}")
    logger.info(f"f-string format: percent={datatypes.Float(0.5):.1%}")

    # Serialize to pyarrow and back
    serialization_start_time = time.perf_counter()
    serialized = datatypes.serialize(my_float)
    serialization_end_time = time.perf_counter()

    deserialization_start_time = time.perf_counter()
    deserialized = datatypes.deserialize(serialized)["param_0"]
    deserialization_end_time = time.perf_counter()

    logger.info(f"Deserialized Float matches Original: {deserialized == my_float}")

    logger.info(
        f"Serialization time: {(serialization_end_time - serialization_start_time) * 1000} ms"
    )
    logger.info(
        f"Deserialization time: {(deserialization_end_time - deserialization_start_time) * 1000} ms"
    )


if __name__ == "__main__":
    float_example()
