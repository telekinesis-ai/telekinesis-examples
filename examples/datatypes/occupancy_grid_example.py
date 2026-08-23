"""Demonstrates the Telekinesis OccupancyGrid datatype."""

import time

import numpy as np
import rerun as rr
from loguru import logger

from telekinesis import datatypes

def occupancy_grid_example():
    """Demonstrate creation, inspection, operations, visualization, and serialization."""

    # ======================= Create ============================================
    height, width = 20, 20

    data = np.full((height, width), datatypes.OccupancyGrid.UNKNOWN, dtype=np.int8)
    data[10, 5] = datatypes.OccupancyGrid.OCCUPIED
    data[10, 6] = datatypes.OccupancyGrid.FREE
    data[10, 7] = datatypes.OccupancyGrid.OCCUPIED

    grid = datatypes.OccupancyGrid(
        data, resolution=0.05, origin_x=-5.0, origin_y=-5.0, origin_yaw=0.0
    )
    logger.info(f"Created OccupancyGrid: {grid}")

    # ======================= Inspect ===========================================
    logger.info(f"data=\n{grid.data}")
    logger.info(f"shape={grid.shape}")
    logger.info(f"height={grid.height}")
    logger.info(f"width={grid.width}")
    logger.info(f"resolution={grid.resolution} m/cell")
    logger.info(f"origin_x={grid.origin_x}")
    logger.info(f"origin_y={grid.origin_y}")
    logger.info(f"origin_yaw={grid.origin_yaw}")
    logger.info(f"FREE={datatypes.OccupancyGrid.FREE}")
    logger.info(f"OCCUPIED={datatypes.OccupancyGrid.OCCUPIED}")
    logger.info(f"UNKNOWN={datatypes.OccupancyGrid.UNKNOWN}")

    # ======================= Operations =========================================
    occupied = int(np.sum(grid.data == datatypes.OccupancyGrid.OCCUPIED))
    free = int(np.sum(grid.data == datatypes.OccupancyGrid.FREE))
    unknown = int(np.sum(grid.data == datatypes.OccupancyGrid.UNKNOWN))
    logger.info(f"occupied={occupied}, free={free}, unknown={unknown}")

    numpy_array = np.asarray(grid)
    logger.info(f"NumPy array via __array__: shape={numpy_array.shape}, dtype={numpy_array.dtype}")

    logger.info(f"grid == grid: {grid == grid}")

    # ======================= Visualize =========================================
    rr.init("occupancy_grid_example", spawn=True)
    datatypes.visualize(grid, entity_path="/occupancy_grid", label="Occupancy Grid")

    # ======================= Serialize / Deserialize ===========================
    start = time.perf_counter()
    serialized = datatypes.serialize(grid)
    serialization_ms = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    deserialized = datatypes.deserialize(serialized)["param_0"]
    deserialization_ms = (time.perf_counter() - start) * 1000

    logger.info(f"Deserialized OccupancyGrid: {deserialized}")
    logger.info(f"Round-trip successful: {grid == deserialized}")
    logger.info(f"Serialization time: {serialization_ms:.3f} ms")
    logger.info(f"Deserialization time: {deserialization_ms:.3f} ms")


if __name__ == "__main__":
    occupancy_grid_example()
