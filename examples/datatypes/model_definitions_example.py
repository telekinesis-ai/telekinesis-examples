"""
Example script to demonstrate usage of the ModelDefinitions datatype.

A `ModelDefinitions` batch holds N model records as parallel columns
(model_names, model_formats, visibilities, ...) rather than a list of
per-model objects -- the natural container for a model listing response.

Shows:
  - constructing a batch from parallel per-field lists
  - accessing batch-level properties (numpy arrays / lists)
  - indexing and slicing the batch
  - round-trip via `serialize` / `deserialize`
  - empty batch edge case
"""

import time
from datetime import datetime, timezone

import rerun as rr
from loguru import logger

from telekinesis import datatypes


def model_definitions_example():
    """
    Example function to demonstrate usage of the ModelDefinitions datatype.
        - Build a batch of two model records
        - Access batch-level properties (numpy arrays / lists)
        - Visualize the batch using Rerun
        - Index a single record, a slice, and a boolean mask
        - Create an empty batch
        - Serialize to PyArrow and back
    """
    dt_a = datetime(2024, 6, 1, tzinfo=timezone.utc)
    dt_b = datetime(2024, 6, 15, tzinfo=timezone.utc)

    # ----- Build the batch -----
    images_input_1 = datatypes.ModelTensorDefinition(
        name="images",
        canonical_name="input_0",  # It has to be input_0
        dtype="float32",
        shape=[1, 3, 224, 224],
    )
    output_0 = datatypes.ModelTensorDefinition(
        name="logits",
        canonical_name="output_0",  # It has to be output_0, output_1, output_2, etc.
        dtype="float32",
        shape=[1, 1000],
    )
    output_1 = datatypes.ModelTensorDefinition(
        name="logits",
        canonical_name="output_0",
        dtype="float32",
        shape=[1, 1000],
    )

    my_model_definitions = datatypes.ModelDefinitions(
        model_names=["model-a", "model-b"],
        model_formats=["onnx", "pytorch"],
        visibilities=["private", "public"],
        model_statuses=["uploaded", "deploying"],
        model_descriptions=["first model", None],
        model_inputs=[[images_input_1], None],
        model_outputs=[[output_0, output_1], None],
        created_ats=[dt_a, dt_b],
        updated_ats=[dt_a, dt_b],
    )

    my_model_definitions_len = len(my_model_definitions)
    my_model_definitions_model_names = my_model_definitions.model_names
    my_model_definitions_model_formats = my_model_definitions.model_formats
    my_model_definitions_visibilities = my_model_definitions.visibilities
    my_model_definitions_model_statuses = my_model_definitions.model_statuses
    my_model_definitions_model_inputs = my_model_definitions.model_inputs
    my_model_definitions_created_ats = my_model_definitions.created_ats
    my_model_definitions_updated_ats = my_model_definitions.updated_ats

    rr.init("model_definitions_example", spawn=True)
    datatypes.visualize(my_model_definitions, entity_path="/ModelDefinitions")

    logger.info(f"Number of ModelDefinitions: {my_model_definitions_len} records")
    logger.info(f"Underlying ModelDefintions model_names:    {my_model_definitions_model_names}")
    logger.info(f"Underlying ModelDefintions model_formats:  {my_model_definitions_model_formats}")
    logger.info(f"Underlying ModelDefintions visibilities:   {my_model_definitions_visibilities}")
    logger.info(f"Underlying ModelDefintions model_statuses: {my_model_definitions_model_statuses}")
    logger.info(f"Underlying ModelDefintions model_inputs:   {my_model_definitions_model_inputs}")
    logger.info(f"Underlying ModelDefintions created_ats:    {my_model_definitions_created_ats}")
    logger.info(f"Underlying ModelDefintions updated_ats:    {my_model_definitions_updated_ats}")

    # Index a single record, a slice, and a boolean mask
    my_first = my_model_definitions[0]
    logger.info(f"My first indexed ModelDefinitions: {my_first}")
    my_subset = my_model_definitions[0:1]
    logger.info(
        f"My sliced ModelDefinitions [0:1]: {len(my_subset)} record(s), names={my_subset.model_names}"
    )
    my_mask = my_model_definitions.model_statuses == "uploaded"
    uploaded_only = my_model_definitions[my_mask]
    logger.info(
        f"My indexed ModelDefinitions with mask[uploaded mask]: {len(uploaded_only)} record(s)"
    )

    # Create an empty batch
    my_empty_model_definitions = datatypes.ModelDefinitions(
        model_names=[], model_formats=[], visibilities=[], model_statuses=[]
    )
    datatypes.visualize(my_empty_model_definitions, entity_path="/ModelDefinitions/empty")

    # Serialize to PyArrow and back
    serialization_start_time = time.perf_counter()
    serialized = datatypes.serialize(my_model_definitions)
    serialization_end_time = time.perf_counter()

    deserialization_start_time = time.perf_counter()
    deserialized = datatypes.deserialize(serialized)["param_0"]
    deserialization_end_time = time.perf_counter()
    logger.info(f"Deserialized ModelDefinitions: {deserialized}")
    logger.info(
        f"Deserialized ModelDefinitions and original batch are equal: {deserialized == my_model_definitions}"
    )

    logger.info(
        f"Serialization time: {(serialization_end_time - serialization_start_time) * 1000} ms"
    )
    logger.info(
        f"Deserialization time: {(deserialization_end_time - deserialization_start_time) * 1000} ms"
    )


if __name__ == "__main__":
    model_definitions_example()
