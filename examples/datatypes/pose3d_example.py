"""
Example script to demonstrate usage of Pose3D datatype.
"""

import time

import numpy as np
from loguru import logger
import rerun as rr

from telekinesis import datatypes


def pose3d_example():
    """
    Example function to demonstrate usage of Pose3D datatype.
     - Create a Pose3D data
     - Print the original data
    """
    # Create a Pose3D data
    input_pose3d_data = [0.5, 0.2, 0.5, 0.4619398, 0.1913417, 0.4619398, 0.7325378]
    logger.info(f"Input Pose3D data: {input_pose3d_data}")

    my_pose3d = datatypes.Pose3D(input_pose3d_data)
    logger.info(f"Original Pose3D: {my_pose3d}")

    my_pose3d_data = my_pose3d.data
    my_pose3d_shape = my_pose3d.shape
    my_pose3d_size = my_pose3d.size
    my_pose3d_dtype = my_pose3d.dtype
    my_pose3d_ndim = my_pose3d.ndim
    my_pose3d_numpy = my_pose3d.to_numpy()
    my_pose3d_copy = my_pose3d.copy()

    logger.info(f"Underlying Pose3D data: {my_pose3d_data}")
    logger.info(f"Underlying Pose3D shape: {my_pose3d_shape}")
    logger.info(f"Underlying Pose3D size: {my_pose3d_size}")
    logger.info(f"Underlying Pose3D dtype: {my_pose3d_dtype}")
    logger.info(f"Underlying Pose3D ndim: {my_pose3d_ndim}")
    logger.info(f"Underlying Pose3D as numpy array: {my_pose3d_numpy}")
    logger.info(f"Underlying Pose3D copy: {my_pose3d_copy}")

    logger.info("Visualizing with Rerun...")
    rr.init("pose3d_example", spawn=True)
    datatypes.visualize(my_pose3d, entity_path="/Pose3D", label="my_pose3d")

    # Update the underlying data via the setter
    my_pose3d.data = [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0]
    logger.info(f"Updated Pose3D: {my_pose3d}")
    datatypes.visualize(my_pose3d, entity_path="/Pose3D/updated", label="updated_pose3d")

    # Convert to transformation matrix
    my_pose3d_matrix = my_pose3d.to_transform_matrix()
    logger.info(f"Pose3D as transformation matrix:\n{my_pose3d_matrix}")
    my_pose3d_transform = datatypes.Transform3D(my_pose3d_matrix)
    datatypes.visualize(my_pose3d_transform, entity_path="/Transform3D", label="pose3d_transform")

    # Convert from transformation matrix back to Pose3D
    my_pose3d_from_transform = datatypes.Pose3D.from_transform_matrix(my_pose3d_transform.data)
    logger.info(f"Pose3D from transformation matrix: {my_pose3d_from_transform}")
    logger.info(
        f"Converted back to Pose3D is equal to original: {my_pose3d_from_transform == my_pose3d}"
    )

    # Create a new Pose3D from a pose in a different rotation representation
    # Rotation as "deg"
    my_pose3d_deg = datatypes.Pose3D.from_pose_format(
        [30, 45, 60, 0.4619398, 0.1913417, 0.4619398, 0.7325378], rot_type="deg"
    )
    logger.info(f"Pose3D from pose with rotation in degrees: {my_pose3d_deg}")

    # Rotation as "rad"
    my_pose3d_rad = datatypes.Pose3D.from_pose_format(
        [
            np.radians(30),
            np.radians(45),
            np.radians(60),
            0.4619398,
            0.1913417,
            0.4619398,
            0.7325378,
        ],
        rot_type="rad",
    )
    logger.info(f"Pose3D from pose with rotation in radians: {my_pose3d_rad}")

    # Rotation as "rotvec"
    my_pose3d_rotvec = datatypes.Pose3D.from_pose_format(
        [0.5235988, 0.7853982, 1.0471976, 0.4619398, 0.1913417, 0.4619398, 0.7325378],
        rot_type="rotvec",
    )
    logger.info(f"Pose3D from pose with rotation as rotation vector: {my_pose3d_rotvec}")

    # Get pose as different rotation representations
    pose_as_deg = my_pose3d.convert_pose_format(rot_type="deg")
    logger.info(f"Pose3D as rotation in degrees: {pose_as_deg}")

    pose_as_rad = my_pose3d.convert_pose_format(rot_type="rad")
    logger.info(f"Pose3D as rotation in radians: {pose_as_rad}")

    pose_as_rotvec = my_pose3d.convert_pose_format(rot_type="rotvec")
    logger.info(f"Pose3D as rotation vector: {pose_as_rotvec}")

    # Use with numpy
    my_pose3d_numpy = np.reshape(my_pose3d, (7,))
    logger.info(f"Underlying Pose3D with np.reshape: {my_pose3d_numpy}")

    # Serialize with PyArrow and deserialize back
    serialization_start_time = time.perf_counter()
    serialized_pose3d = datatypes.serialize(my_pose3d)
    serialization_end_time = time.perf_counter()

    deserialization_start_time = time.perf_counter()
    deserialized_pose3d = datatypes.deserialize(serialized_pose3d)["param_0"]
    deserialization_end_time = time.perf_counter()
    logger.info(f"Deserialized Pose3D: {deserialized_pose3d}")
    logger.info(f"Deserialized Pose3D is equal to original: {deserialized_pose3d == my_pose3d}")

    logger.info(f"Serialization time: {serialization_end_time - serialization_start_time} seconds")
    logger.info(
        f"Deserialization time: {deserialization_end_time - deserialization_start_time} seconds"
    )


if __name__ == "__main__":
    pose3d_example()
