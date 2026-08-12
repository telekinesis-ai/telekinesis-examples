"""
Example script to demonstrate usage of Poses3D datatype.
"""

from loguru import logger
import rerun as rr

from telekinesis import datatypes


def poses3d_example():
    """
    Example function to demonstrate usage of Poses3D datatype.
     - Create a Poses3D data
     - Print the original data
    """
    # Create a Poses3D data
    my_poses3d = datatypes.Poses3D(
        [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0]]
    )
    logger.info(f"Original Poses3D: {my_poses3d}")

    # Access the underlying poses data
    my_poses3d_data = my_poses3d.data
    logger.info(f"Underlying Poses3D data: {my_poses3d_data}")

    logger.info("Visualizing with Rerun...")
    rr.init("poses3d_example", spawn=True)
    datatypes.visualize(my_poses3d, entity_path="/Poses3D", label=["My Poses3D 0", "My Poses3D 1"])


if __name__ == "__main__":
    poses3d_example()
