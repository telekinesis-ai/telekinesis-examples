"""
Demonstrates extracting the normal vector from plane coefficients.
"""

from loguru import logger
import rerun as rr

from telekinesis import vitreous, datatypes


def calculate_plane_normal_example():
    """
    Extracts the normal vector from plane coefficients.

    Extracts and normalizes the normal vector from plane equation coefficients
    (ax + by + cz + d = 0).
    """
    # ===================== Run Skill ==========================================
    normal_vector = vitreous.calculate_plane_normal(plane_coefficients=[0.0, 0.0, 1.0, 0.0])

    # ===================== Log ================================================
    logger.success(f"Calculated normal vector to {normal_vector}")
    logger.success(f"Results: {normal_vector}")
    logger.info(f"Normal vector as numpy array: {normal_vector.data}")
    logger.info(f"Normal vector shape: {normal_vector.shape}")
    logger.info(f"Normal vector ndim: {normal_vector.ndim}")
    logger.info(f"Normal vector dtype: {normal_vector.dtype}")

    # ===================== Visualization  (Optional) ===========================
    rr.init("calculate_plane_normal_example", spawn=True)
    datatypes.visualize(normal_vector, entity_path="/PlaneNormal/normal_vector", label="Normal Vector")


if __name__ == "__main__":
    calculate_plane_normal_example()
