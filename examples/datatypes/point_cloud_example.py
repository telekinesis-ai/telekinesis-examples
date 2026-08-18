"""Demonstrates the Telekinesis PointCloud datatype."""

from pathlib import Path
import time

import numpy as np
from loguru import logger
import rerun as rr
import rerun.blueprint as rrb

from telekinesis import datatypes

ROOT_PATH = Path(__file__).parent.parent

def point_cloud_example():
    """Demonstrate creation of compressed and uncompressed point clouds, access, visualization, update, compression settings, loading/saving .ply files, NumPy interop, and serialization."""

    # ======================= Create ============================================
    N = 4000000
    positions = np.random.randn(N, 3).astype(np.float32)
    normals = np.random.randn(N, 3).astype(np.float32)
    colors = np.random.randint(0, 255, (N, 3), dtype=np.uint8)

    uncompressed = datatypes.PointCloud(
        positions=positions, normals=normals, colors=colors, use_compression=False
    )
    compressed = datatypes.PointCloud(
        positions=positions, normals=normals, colors=colors, use_compression=True
    )
    logger.info(f"Original Uncompressed PointCloud: {uncompressed}")
    logger.info(f"Original Compressed PointCloud: {compressed}")

    # ======================= Inspect ===========================================
    positions = uncompressed.positions
    normals = uncompressed.normals
    colors = uncompressed.colors
    compression_settings = uncompressed.compression_settings

    logger.info(f"Underlying positions: {positions}")
    logger.info(f"Underlying normals: {normals}")
    logger.info(f"Underlying colors: {colors}")
    logger.info(f"Compression settings: {compression_settings}")

    # ======================= Visualize =========================================
    blueprint = rrb.Blueprint(
        rrb.Grid(
            rrb.Spatial3DView(name="My PointCloud", origin="/PointCloud/my_pointcloud"),
            rrb.Spatial3DView(name="Updated PointCloud", origin="/PointCloud/updated"),
            rrb.Spatial3DView(
                name="Updated PointCloud Colors", origin="/PointCloud/updated_colors"
            ),
        )
    )
    rr.init("point_cloud_example", spawn=True, default_blueprint=blueprint)
    rr.send_blueprint(blueprint, make_active=True)
    datatypes.visualize(
        uncompressed, entity_path="/PointCloud/my_pointcloud", label="My PointCloud"
    )

    # ======================= Update ============================================
    updated_positions = np.random.randn(N, 3).astype(np.float32)
    uncompressed.positions = updated_positions
    logger.info(f"Updated positions: {uncompressed}")

    updated_colors = np.random.randint(0, 255, (N, 3), dtype=np.uint8)
    uncompressed.colors = updated_colors
    logger.info(f"Updated colors: {uncompressed.colors}")
    datatypes.visualize(
        uncompressed,
        entity_path="/PointCloud/updated_colors",
        label="Updated PointCloud Colors",
    )

    # ======================= Compression =======================================
    uncompressed.use_compression = True
    uncompressed.set_compression_parameters(compression_level=5, quantization_bits=12)
    logger.info(f"Updated compression settings: {uncompressed}")

    # ======================= Load / Save =======================================
    url = "https://assets.telekinesis.ai/examples/v1/point_clouds/zivid_bottles_10_preprocessed.ply"
    from_url = datatypes.PointCloud.from_url(url=url)
    logger.info(f"PointCloud from .ply: {from_url}")
    datatypes.visualize(from_url, entity_path="/PointCloud/from_url", label="URL PointCloud")

    from_url.save_to_path("results/my_point_cloud_saved.ply")
    logger.info("Saved PointCloud to disk as .ply file.")

    # ======================= NumPy Interop =====================================
    numpy_data = uncompressed.to_numpy()
    array_data = np.asarray(uncompressed)
    centroid = np.mean(uncompressed, axis=0)
    logger.info(f"NumPy array: {numpy_data}")
    logger.info(f"As array: {array_data}")
    logger.info(f"Centroid: {centroid}")

    # ======================= Serialize / Deserialize ===========================
    start = time.perf_counter()
    serialized_uncompressed = datatypes.serialize(uncompressed)
    uncompressed_serialization_ms = (time.perf_counter() - start) * 1000
    uncompressed_serialized_size = len(serialized_uncompressed)

    start = time.perf_counter()
    deserialized = datatypes.deserialize(serialized_uncompressed)["param_0"]
    uncompressed_deserialization_ms = (time.perf_counter() - start) * 1000
    uncompressed_deserialized_size = len(serialized_uncompressed)

    logger.info(f"Deserialized Uncompressed PointCloud: {deserialized}")
    logger.info(f"Round-trip successful: {deserialized == uncompressed}")

    start = time.perf_counter()
    serialized_compressed = datatypes.serialize(compressed)
    compressed_serialization_ms = (time.perf_counter() - start) * 1000
    compressed_serialized_size = len(serialized_compressed)

    start = time.perf_counter()
    deserialized = datatypes.deserialize(serialized_compressed)["param_0"]
    compressed_deserialization_ms = (time.perf_counter() - start) * 1000
    compressed_deserialized_size = len(serialized_compressed)

    logger.info(f"Deserialized Compressed PointCloud: {deserialized}")
    logger.info(f"Round-trip successful: {deserialized == compressed}")

    logger.info(
        f"Uncompressed: serialize={uncompressed_serialization_ms:.3f} ms "
        f"({uncompressed_serialized_size / 1024 / 1024:.3f} MB), "
        f"deserialize={uncompressed_deserialization_ms:.3f} ms "
        f"({uncompressed_deserialized_size / 1024 / 1024:.3f} MB)"
    )
    logger.info(
        f"Compressed: serialize={compressed_serialization_ms:.3f} ms "
        f"({compressed_serialized_size / 1024 / 1024:.3f} MB), "
        f"deserialize={compressed_deserialization_ms:.3f} ms "
        f"({compressed_deserialized_size / 1024 / 1024:.3f} MB)"
    )


if __name__ == "__main__":
    point_cloud_example()
