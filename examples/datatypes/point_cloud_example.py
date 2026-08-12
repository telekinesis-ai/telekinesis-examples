"""
Example script to demonstrate usage of PointCloud datatype.

Shows both serialization paths via the `use_compression`
constructor flag:
  - default (lossless) — `serialize` uses the plain PyArrow layout
  - opt-in (Draco)     — `serialize` uses the Draco-compressed layout
"""

from pathlib import Path
import time

import numpy as np
from loguru import logger
import rerun as rr
import rerun.blueprint as rrb

from telekinesis import datatypes

ROOT_PATH = Path(__file__).parent.parent


def point_cloud_example():
    """
    Example function to demonstrate usage of PointCloud datatype.
     - Build a PointCloud
     - Access positions / normals / colors
     - Visualize the PointCloud data using Rerun
     - Update the PointCloud data
     - Operate with numpy arrays directly via `to_numpy()` / `np.asarray()`
     - Serialize and deserialize the PointCloud
    """
    N = 4000000
    positions = np.random.randn(N, 3).astype(np.float32)
    normals = np.random.randn(N, 3).astype(np.float32)
    colors = np.random.randint(0, 255, (N, 3), dtype=np.uint8)

    # Build a PointCloud (default lossless path)
    my_uncompressed_point_cloud = datatypes.PointCloud(
        positions=positions, normals=normals, colors=colors, use_compression=False
    )
    my_compressed_point_cloud = datatypes.PointCloud(
        positions=positions, normals=normals, colors=colors, use_compression=True
    )
    logger.info(f"Original Uncompressed PointCloud: {my_uncompressed_point_cloud}")
    logger.info(f"Original Compressed PointCloud: {my_compressed_point_cloud}")

    # Access underlying positions / normals / colors
    my_point_cloud_positions = my_uncompressed_point_cloud.positions
    my_point_cloud_normals = my_uncompressed_point_cloud.normals
    my_point_cloud_colors = my_uncompressed_point_cloud.colors
    my_point_cloud_compression_settings = my_uncompressed_point_cloud.compression_settings

    logger.info(f"Underlying PointCloud positions: {my_point_cloud_positions}")
    logger.info(f"Underlying PointCloud normals: {my_point_cloud_normals}")
    logger.info(f"Underlying PointCloud colors: {my_point_cloud_colors}")
    logger.info(f"PointCloud compression settings: {my_point_cloud_compression_settings}")

    # Give each PointCloud its own tile instead of merging them into one shared
    # 3D view, since they'd otherwise overlap (all are random points near the origin).
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
    # `default_blueprint` above only applies if the viewer has no active blueprint yet --
    # if a viewer from a previous run is still open, it already has one active, so force it.
    rr.send_blueprint(blueprint, make_active=True)
    datatypes.visualize(
        my_uncompressed_point_cloud, entity_path="/PointCloud/my_pointcloud", label="My PointCloud"
    )

    # Update the PointCloud data
    updated_positions = np.random.randn(N, 3).astype(np.float32)
    my_uncompressed_point_cloud.positions = updated_positions
    logger.info(f"Updated PointCloud: {my_uncompressed_point_cloud}")
    datatypes.visualize(
        my_uncompressed_point_cloud, entity_path="/PointCloud/updated", label="Updated PointCloud"
    )

    updated_colors = np.random.randint(0, 255, (N, 3), dtype=np.uint8)
    my_uncompressed_point_cloud.colors = updated_colors
    logger.info(f"Updated PointCloud colors: {my_uncompressed_point_cloud.colors}")
    datatypes.visualize(
        my_uncompressed_point_cloud,
        entity_path="/PointCloud/updated_colors",
        label="Updated PointCloud Colors",
    )

    # Update the compression settings for the uncompressed PointCloud
    my_uncompressed_point_cloud.use_compression = True
    my_uncompressed_point_cloud.set_compression_parameters(
        compression_level=5, quantization_bits=12
    )
    logger.info(f"Updated PointCloud compression settings: {my_uncompressed_point_cloud}")

    # Create from .ply file from path
    my_point_cloud_from_ply = datatypes.PointCloud.from_path(ROOT_PATH / "data/my_point_cloud.ply")
    logger.info(f"My new PointCloud from .ply: {my_point_cloud_from_ply}")

    # Save to .ply file to disk
    my_uncompressed_point_cloud.save_to_path("results/my_point_cloud_saved.ply")
    logger.info("Saved PointCloud to disk as .ply file.")

    # Operate with numpy arrays directly
    my_point_cloud_numpy = my_uncompressed_point_cloud.to_numpy()
    logger.info(f"My Uncompressed PointCloud to numpy: {my_point_cloud_numpy}")

    my_point_cloud_asarray = np.asarray(my_uncompressed_point_cloud)
    logger.info(f"My Uncompressed PointCloud as array: {my_point_cloud_asarray}")

    centroid = np.mean(my_uncompressed_point_cloud, axis=0)
    logger.info(f"Centroid of the My Uncompressed PointCloud: {centroid}")

    # Serialize to PyArrow and back
    serialization_start_time = time.perf_counter()
    serialized_uncompressed_point_cloud = datatypes.serialize(my_uncompressed_point_cloud)
    serialization_end_time = time.perf_counter()
    uncompressed_serialized_size = len(serialized_uncompressed_point_cloud)
    uncompressed_serialized_time = (serialization_end_time - serialization_start_time) * 1000

    deserialization_start_time = time.perf_counter()
    deserialized_point_cloud = datatypes.deserialize(serialized_uncompressed_point_cloud)["param_0"]
    deserialization_end_time = time.perf_counter()
    uncompressed_deserialized_size = len(serialized_uncompressed_point_cloud)
    uncompressed_deserialized_time = (deserialization_end_time - deserialization_start_time) * 1000
    logger.info(f"Deserialized Uncompressed PointCloud: {deserialized_point_cloud}")
    logger.info(
        f"Deserialized Uncompressed PointCloud matches original: {deserialized_point_cloud == my_uncompressed_point_cloud}"
    )

    # For the compressed PointCloud, this will use the Draco-compressed path.
    serialization_start_time = time.perf_counter()
    serialized_compressed_point_cloud = datatypes.serialize(my_compressed_point_cloud)
    serialization_end_time = time.perf_counter()
    compressed_serialized_size = len(serialized_compressed_point_cloud)
    compressed_serialized_time = (serialization_end_time - serialization_start_time) * 1000

    deserialization_start_time = time.perf_counter()
    deserialized_point_cloud = datatypes.deserialize(serialized_compressed_point_cloud)["param_0"]
    deserialization_end_time = time.perf_counter()
    compressed_deserialized_size = len(serialized_compressed_point_cloud)
    compressed_deserialized_time = (deserialization_end_time - deserialization_start_time) * 1000

    logger.info(f"Uncompressed Serialization time: {uncompressed_serialized_time:.6f} ms")
    logger.info(f"Compressed Serialization time: {compressed_serialized_time:.6f} ms")

    logger.info(f"Uncompressed Serialized size: {uncompressed_serialized_size / 1024 / 1024} MB")
    logger.info(f"Compressed Serialized size: {compressed_serialized_size / 1024 / 1024} MB")

    logger.info(f"Uncompressed Deserialization time: {uncompressed_deserialized_time:.6f} ms")
    logger.info(f"Compressed Deserialization time: {compressed_deserialized_time:.6f} ms")

    logger.info(
        f"Uncompressed Deserialized size: {uncompressed_deserialized_size / 1024 / 1024} MB"
    )
    logger.info(f"Compressed Deserialized size: {compressed_deserialized_size / 1024 / 1024} MB")


if __name__ == "__main__":
    point_cloud_example()
