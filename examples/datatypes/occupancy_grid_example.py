"""Demonstrates the Telekinesis OccupancyGrid datatype."""

import time

import numpy as np
from loguru import logger

from telekinesis import datatypes

def occupancy_grid_example():
    """Demonstrate creation with the FREE/OCCUPIED/UNKNOWN constants, access, and serialization."""

    # ======================= Create ============================================
    height, width = 20, 20

    data = np.full((height, width), datatypes.OccupancyGrid.UNKNOWN, dtype=np.int8)
    data[10, 5] = datatypes.OccupancyGrid.OCCUPIED
    data[10, 6] = datatypes.OccupancyGrid.FREE
    data[10, 7] = datatypes.OccupancyGrid.OCCUPIED

    grid = datatypes.OccupancyGrid(data, resolution=0.05, origin_x=-5.0, origin_y=-5.0, origin_yaw=0.0)

    logger.info(f"Created OccupancyGrid: {grid}")

    # ======================= Inspect ===========================================
    logger.info(
        f"shape={grid.shape}, height={grid.height}, width={grid.width}, "
        f"resolution={grid.resolution} m/cell"
    )
    logger.info(f"origin_x={grid.origin_x}, origin_y={grid.origin_y}, origin_yaw={grid.origin_yaw}")

    occupied = int(np.sum(grid.data == datatypes.OccupancyGrid.OCCUPIED))
    free = int(np.sum(grid.data == datatypes.OccupancyGrid.FREE))
    unknown = int(np.sum(grid.data == datatypes.OccupancyGrid.UNKNOWN))

    logger.info(f"occupied={occupied}, free={free}, unknown={unknown}")

    # ======================= Serialize / Deserialize ===========================
    start = time.perf_counter()
    serialized = datatypes.serialize(grid)
    serialization_ms = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    deserialized = datatypes.deserialize(serialized)["param_0"]
    deserialization_ms = (time.perf_counter() - start) * 1000

    assert grid == deserialized, "round-trip mismatch"

    logger.info(f"Deserialized OccupancyGrid: {deserialized}")
    logger.info(f"Round-trip successful: {grid == deserialized}")
    logger.info(f"Serialization time: {serialization_ms:.3f} ms")
    logger.info(f"Deserialization time: {deserialization_ms:.3f} ms")


if __name__ == "__main__":
    occupancy_grid_example()
