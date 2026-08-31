"""Demonstrates the Telekinesis USD datatype."""

import time

from loguru import logger

from telekinesis import datatypes


def usd_example():
    """Demonstrate fetching, inspection, and serialization."""

    # ======================= Create ==============================================
    # The archive holds one .usd at its top level, plus whatever layers that
    # stage references, so the stage opens straight from the extracted folder.
    usd = datatypes.USD.from_url(
        "https://assets.telekinesis.ai/usd/tools/welding_guns/spot_welding_gun.zip"
    )
    logger.info(f"Fetched USD: {usd}")

    # ======================= Inspect ============================================
    logger.info(f"path={usd.path}")

    # ======================= Serialize / Deserialize ============================
    start = time.perf_counter()
    serialized = datatypes.serialize(usd)
    serialization_ms = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    deserialized = datatypes.deserialize(serialized)["param_0"]
    deserialization_ms = (time.perf_counter() - start) * 1000

    logger.info(f"Deserialized USD: {deserialized}")
    logger.info(f"Round-trip successful: {usd == deserialized}")
    logger.info(f"Serialization time: {serialization_ms:.3f} ms")
    logger.info(f"Deserialization time: {deserialization_ms:.3f} ms")


if __name__ == "__main__":
    usd_example()
