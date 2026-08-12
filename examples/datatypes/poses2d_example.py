"""
Example script to demonstrate usage of Poses2D datatype.
"""

from loguru import logger
import rerun as rr

from telekinesis import datatypes


def poses2d_example():
    """
    Example function to demonstrate usage of Poses2D datatype.
     - Create a Poses2D data
     - Print the original data
    """
    # Create a Poses2D data
    my_poses2d = datatypes.Poses2D([[1.0, 2.0, 0.5], [3.0, 4.0, 1.2]])
    logger.info(f"Original Poses2D: {my_poses2d}")

    # Access the underlying poses data
    my_poses2d_data = my_poses2d.data
    logger.info(f"Underlying Poses2D data: {my_poses2d_data}")

    logger.info("Visualizing with Rerun...")
    rr.init("poses2d_example", spawn=True)
    datatypes.visualize(my_poses2d, entity_path="/Poses2D", label=["My Poses2D 0", "My Poses2D 1"])


if __name__ == "__main__":
    poses2d_example()
