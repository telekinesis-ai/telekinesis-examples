"""Demonstrates the Telekinesis ModelDefinitions datatype."""

import time
from datetime import datetime, timezone

import rerun as rr
from loguru import logger

from telekinesis import datatypes

def model_definitions_example():
    """Demonstrate creation, inspection, operations, visualization, and serialization."""

    # ======================= Create ============================================
    created_at = datetime(2024, 6, 1, tzinfo=timezone.utc)
    updated_at = datetime(2024, 6, 15, tzinfo=timezone.utc)

    model_input = datatypes.ModelTensorDefinition(
        name="images", canonical_name="input_0", dtype="float32", shape=[1, 3, 224, 224]
    )
    model_output = datatypes.ModelTensorDefinition(
        name="logits", canonical_name="output_0", dtype="float32", shape=[1, 1000]
    )
    other_output = datatypes.ModelTensorDefinition(
        name="logits", canonical_name="output_0", dtype="float32", shape=[1, 1000]
    )

    definitions = datatypes.ModelDefinitions(
        model_names=["model-a", "model-b"],
        model_formats=["onnx", "pytorch"],
        visibilities=["private", "public"],
        model_statuses=["uploaded", "deploying"],
        model_descriptions=["first model", None],
        model_inputs=[[model_input], None],
        model_outputs=[[model_output, other_output], None],
        created_ats=[created_at, updated_at],
        updated_ats=[created_at, updated_at],
    )
    logger.info(f"Created ModelDefinitions: {definitions}")

    # ======================= Inspect ===========================================
    logger.info(f"model_names={definitions.model_names}")
    logger.info(f"model_formats={definitions.model_formats}")
    logger.info(f"visibilities={definitions.visibilities}")
    logger.info(f"model_statuses={definitions.model_statuses}")
    logger.info(f"model_descriptions={definitions.model_descriptions}")
    logger.info(f"model_inputs={definitions.model_inputs}")
    logger.info(f"model_outputs={definitions.model_outputs}")
    logger.info(f"created_ats={definitions.created_ats}")
    logger.info(f"updated_ats={definitions.updated_ats}")

    # ======================= Operations =========================================
    logger.info(f"length={len(definitions)}")

    subset = definitions[0:1]
    logger.info(f"definitions[0:1] = {len(subset)} record(s), names={subset.model_names}")

    first = definitions[0]
    logger.info(f"definitions[0] = {first}")

    mask = definitions.model_statuses == "uploaded"
    logger.info(f"definitions[uploaded mask] = {len(definitions[mask])} record(s)")

    # ModelTensorDefinition exposes its fields as plain attributes plus a dict view.
    logger.info(f"model_input.to_dict()={model_input.to_dict()}")

    # A ModelDefinitions batch may be empty.
    empty = datatypes.ModelDefinitions(
        model_names=[], model_formats=[], visibilities=[], model_statuses=[]
    )
    logger.info(f"length of empty batch={len(empty)}")

    # ======================= Visualize =========================================
    rr.init("model_definitions_example", spawn=True)
    datatypes.visualize(definitions, entity_path="/model_definitions")
    datatypes.visualize(empty, entity_path="/model_definitions/empty")

    # ======================= Serialize / Deserialize ===========================
    start = time.perf_counter()
    serialized = datatypes.serialize(definitions)
    serialization_ms = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    deserialized = datatypes.deserialize(serialized)["param_0"]
    deserialization_ms = (time.perf_counter() - start) * 1000

    logger.info(f"Deserialized ModelDefinitions: {deserialized}")
    logger.info(f"Round-trip successful: {definitions == deserialized}")
    logger.info(f"Serialization time: {serialization_ms:.3f} ms")
    logger.info(f"Deserialization time: {deserialization_ms:.3f} ms")


if __name__ == "__main__":
    model_definitions_example()
