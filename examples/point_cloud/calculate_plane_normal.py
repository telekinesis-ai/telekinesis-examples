"""
Demonstrates extracting the normal vector from plane coefficients.

This example:
- Extracts and normalizes the normal vector from plane equation coefficients.
- Visualizes the result using Rerun.
"""

from loguru import logger
import rerun as rr

from telekinesis import vitreous, datatypes


def calculate_plane_normal_example():
    """
    Extracts the normal vector from plane coefficients.

    Demonstrates extracting and normalizing the normal vector from plane equation
    coefficients (ax + by + cz + d = 0).
    """
    # ===================== Run Skill ==========================================
    normal_vector = vitreous.calculate_plane_normal(plane_coefficients=[0.0, 0.0, 1.0, 0.0])
    logger.success(
        f"Calculated normal vector to {normal_vector}"
    )
    # Access the data attribute of the normal vector to get the underlying numpy array
    normal_vector_arr = normal_vector.data
    logger.info(f"Normal vector as numpy array: {normal_vector_arr}")

    normal_vector_shape = normal_vector.shape
    normal_vector_ndim = normal_vector.ndim
    normal_vector_dtype = normal_vector.dtype
    logger.info(f"Normal vector shape: {normal_vector_shape}")
    logger.info(f"Normal vector ndim: {normal_vector_ndim}")
    logger.info(f"Normal vector dtype: {normal_vector_dtype}")

    # ===================== Visualization  (Optional) ======================
    rr.init("calculate_plane_normal_example", spawn=True)
    datatypes.visualize(normal_vector, entity_path="/PlaneNormal/normal_vector", label="Normal Vector")


if __name__ == "__main__":
    calculate_plane_normal_example()
