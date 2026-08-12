"""
Example script to demonstrate usage of the OccupancyGrid datatype.

An `OccupancyGrid` holds a 2D `(H, W)` `int8` grid mapping world cells
to occupancy status using the ROS nav_msgs/OccupancyGrid three-value
convention:
  - FREE     (0)   -- cell is known to be unoccupied
  - OCCUPIED (100) -- cell is known to be occupied
  - UNKNOWN  (-1)  -- cell has not been observed yet

Shows:
  - constructing a grid from an int8 array with resolution and origin
  - using the FREE / OCCUPIED / UNKNOWN class constants
  - accessing individual properties
  - round-trip via `serialize` / `deserialize`
"""

import numpy as np
from loguru import logger

from telekinesis import datatypes


def occupancy_grid_example():
    """
    Example function to demonstrate usage of the OccupancyGrid datatype.
     - Build a grid with known free, occupied, and unknown cells
     - Access individual properties
     - Round-trip via serialize / deserialize
    """
    H, W = 20, 20

    # Start with all cells unknown, then mark a few explicitly.
    data = np.full((H, W), datatypes.OccupancyGrid.UNKNOWN, dtype=np.int8)
    data[10, 5] = datatypes.OccupancyGrid.OCCUPIED
    data[10, 6] = datatypes.OccupancyGrid.FREE
    data[10, 7] = datatypes.OccupancyGrid.OCCUPIED

    grid = datatypes.OccupancyGrid(
        data,
        resolution=0.05,  # 5 cm per cell
        origin_x=-5.0,
        origin_y=-5.0,
        origin_yaw=0.0,
    )

    logger.info(f"OccupancyGrid: {grid}")
    logger.info(f"  shape:       {grid.shape}")
    logger.info(f"  height:      {grid.height}")
    logger.info(f"  width:       {grid.width}")
    logger.info(f"  resolution:  {grid.resolution} m/cell")
    logger.info(f"  origin_x:    {grid.origin_x}")
    logger.info(f"  origin_y:    {grid.origin_y}")
    logger.info(f"  origin_yaw:  {grid.origin_yaw}")

    occupied_cells = int(np.sum(grid.data == datatypes.OccupancyGrid.OCCUPIED))
    free_cells = int(np.sum(grid.data == datatypes.OccupancyGrid.FREE))
    unknown_cells = int(np.sum(grid.data == datatypes.OccupancyGrid.UNKNOWN))
    logger.info(f"  occupied:    {occupied_cells} cells")
    logger.info(f"  free:        {free_cells} cells")
    logger.info(f"  unknown:     {unknown_cells} cells")

    # ----- Round-trip -----
    serialized = datatypes.serialize(grid)
    restored = datatypes.deserialize(serialized)["param_0"]
    assert grid == restored, "round-trip mismatch"
    logger.info(f"Round-trip restored: {restored}")


if __name__ == "__main__":
    occupancy_grid_example()
