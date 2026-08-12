"""
Example script to demonstrate usage of Transform3D datatype.
"""

import time

import numpy as np
from loguru import logger
import rerun as rr

from telekinesis import datatypes


def transform3d_example():
    """
    Example function to demonstrate usage of Transform3D datatype.
        - Create a Transform3D data (4x4 homogeneous transform)
        - Access the underlying transform data
        - Visualize the Transform3D data using Rerun
        - Update the underlying transform data
        - Convert to pose numpy array
        - Calculate the inverse of the transform
        - Operate on the underlying data with numpy
        - Convert back to Transform3D from pose numpy array
        - Serialize to PyArrow and back
    """
    # Create a Transform3D data (4x4 homogeneous transform;)
    matrix = np.array(
        [
            [0.5000000, -0.5000000, 0.7071068, 1],
            [0.8535534, 0.1464466, -0.5000000, 2],
            [0.1464466, 0.8535534, 0.5000000, 3],
            [0, 0, 0, 1],
        ]
    )
    my_transform3d = datatypes.Transform3D(matrix)
    logger.info(f"Original Transform3D: {my_transform3d}")

    # Access the underlying transform data
    my_transform3d_data = my_transform3d.data
    my_transform3d_shape = my_transform3d.shape
    my_transform3d_size = my_transform3d.size
    my_transform3d_dtype = my_transform3d.dtype
    my_transform3d_ndim = my_transform3d.ndim
    my_transform3d_numpy = my_transform3d.to_numpy()
    my_transform3d_copy = my_transform3d.copy()

    logger.info(f"Underlying Transform3D data: {my_transform3d_data}")
    logger.info(f"Underlying Transform3D data shape: {my_transform3d_shape}")
    logger.info(f"Underlying Transform3D data size: {my_transform3d_size}")
    logger.info(f"Underlying Transform3D data dtype: {my_transform3d_dtype}")
    logger.info(f"Underlying Transform3D data ndim: {my_transform3d_ndim}")
    logger.info(f"Underlying Transform3D data as numpy array: {my_transform3d_numpy}")
    logger.info(f"Underlying Transform3D object: {my_transform3d_copy}")

    logger.info("Visualizing with Rerun...")
    rr.init("transform3d_example", spawn=True)
    datatypes.visualize(my_transform3d, entity_path="/Transform3D/main", label="my_transform3d")

    # Update the my_transform3d_data
    new_matrix = np.array(
        [
            [0.5000000, -0.5000000, 0.7071068, 1.5],
            [0.8535534, 0.1464466, -0.5000000, 2.5],
            [0.1464466, 0.8535534, 0.5000000, 3],
            [0, 0, 0, 1],
        ]
    )
    my_transform3d.data = new_matrix
    logger.info(f"Updated Transform3D: {my_transform3d}")
    datatypes.visualize(
        my_transform3d, entity_path="/Transform3D/updated", label="updated_transform3d"
    )

    # Calculate the inverse of the transform
    inverse_transform3d_matrix = my_transform3d.inverse()
    logger.info(f"Inverse Transform3D: {inverse_transform3d_matrix}")
    inverse_transform3d = datatypes.Transform3D(inverse_transform3d_matrix)
    datatypes.visualize(
        inverse_transform3d, entity_path="/Transform3D/inverse", label="inverse_transform3d"
    )

    # Convert to pose numpy array with rotation type as degrees
    pose_array = my_transform3d.to_pose(rot_type="deg")
    logger.info(f"Pose numpy array with rot_type='deg': {pose_array}")

    # Convert to pose numpy array with rotation type as rotvec
    pose_array = my_transform3d.to_pose(rot_type="rotvec")
    logger.info(f"Pose numpy array with rot_type='rotvec': {pose_array}")

    # Convert to pose numpy array with rotation type as rad
    pose_array = my_transform3d.to_pose(rot_type="rad")
    logger.info(f"Pose numpy array with rot_type='rad': {pose_array}")

    # Convert to pose numpy array with quaternion rotation type
    # Note: The quaternion is returned in the format [qx, qy, qz, qw] (scalar first)
    # Above quaternion format is followed throughout the Telekinesis library for consistency.
    pose_array = my_transform3d.to_pose(rot_type="quat")
    logger.info(f"Pose numpy array with rot_type='quat': {pose_array}")

    # Convert back to Transform3D from pose numpy array
    new_transform3d = datatypes.Transform3D.from_pose(pose_array)
    logger.info(f"New Transform3D from pose: {new_transform3d}")

    # Check if new transform is equal to original transform
    # using the compute_transformation_error method
    my_transform3d_error = my_transform3d.compute_transformation_error(new_transform3d)
    logger.info(f"Transformation error between original and new transform: {my_transform3d_error}")

    # Operate on the underlying data with numpy - Add to the last column of the transform matrix
    my_transform3d_sum = np.array([1, 1, 1, 0]) + my_transform3d_numpy
    logger.info(f"Sum of Transform3D with numpy array : {my_transform3d_sum}")

    # Serialize to PyArrow and back
    serialization_start_time = time.perf_counter()
    serialized = datatypes.serialize(my_transform3d)
    serialization_end_time = time.perf_counter()

    deserialization_start_time = time.perf_counter()
    deserialized_transform3d = datatypes.deserialize(serialized)["param_0"]
    logger.info(f"Deserialized Transform3D: {deserialized_transform3d}")
    logger.info(
        f"Deserialized Transform3D is equal to original: {deserialized_transform3d == my_transform3d}"
    )
    deserialization_end_time = time.perf_counter()

    logger.info(
        f"Serialization time: {(serialization_end_time - serialization_start_time) * 1000} ms"
    )
    logger.info(
        f"Deserialization time: {(deserialization_end_time - deserialization_start_time) * 1000} ms"
    )


if __name__ == "__main__":
    transform3d_example()
