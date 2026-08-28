"""Demonstrates the Telekinesis URDF datatype.

If https://assets.telekinesis.ai is unreachable, serve a local
`telekinesis-assets` checkout instead and point the URL below at it:

    1. cd /path/to/telekinesis-assets
    2. python3 -m http.server 8931
    3. Replace "https://assets.telekinesis.ai/..." below with
       "http://localhost:8931/..." (same path after that).

Stop the server with Ctrl+C in that terminal when done.
"""

import time

from loguru import logger

from telekinesis import datatypes


def urdf_example():
    """Demonstrate fetching, inspection, and serialization."""

    # ======================= Fetch ==============================================
    # from_url downloads and extracts a .zip bundle, returning a fully
    # populated URDF with its file paths already resolved.
    urdf = datatypes.URDF.from_url(
        # "https://assets.telekinesis.ai/urdf/robots/manipulators/universal_robots/ur10e.zip"
        "http://localhost:8931/urdf/robots/manipulators/universal_robots/ur10e.zip"
    )
    logger.info(f"Fetched URDF: {urdf}")

    # ======================= Inspect ============================================
    logger.info(f"path={urdf.path}")
    logger.info(f"srdf_path={urdf.srdf_path}")
    logger.info(f"mesh_dir={urdf.mesh_dir}")

    # ======================= Serialize / Deserialize ============================
    start = time.perf_counter()
    serialized = datatypes.serialize(urdf)
    serialization_ms = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    deserialized = datatypes.deserialize(serialized)["param_0"]
    deserialization_ms = (time.perf_counter() - start) * 1000

    logger.info(f"Deserialized URDF: {deserialized}")
    logger.info(f"Round-trip successful: {urdf == deserialized}")
    logger.info(f"Serialization time: {serialization_ms:.3f} ms")
    logger.info(f"Deserialization time: {deserialization_ms:.3f} ms")


if __name__ == "__main__":
    urdf_example()
