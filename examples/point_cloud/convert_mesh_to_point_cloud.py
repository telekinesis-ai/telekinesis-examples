"""
Demonstrates sampling a mesh's surface into a 3D point cloud.
"""

import numpy as np
from loguru import logger
import rerun as rr

from telekinesis import vitreous, datatypes


def convert_mesh_to_point_cloud_example():
    """
    Samples a mesh's surface into a point cloud.

    Converts a mesh into a point cloud by sampling points across its surface,
    using Poisson-disk sampling for even ("blue noise") coverage.
    """
    # ===================== Load Data ==========================================
    cylinder_mesh = vitreous.create_cylinder_mesh(
        radius=0.01,
        height=0.02,
        radial_resolution=20,
        height_resolution=4,
        retain_base=False,
        vertex_tolerance=1e-6,
        transformation_matrix=np.eye(4, dtype=np.float32),
        compute_vertex_normals=True,
    )

    # ===================== Run Skill ==========================================
    point_cloud = vitreous.convert_mesh_to_point_cloud(
        mesh=cylinder_mesh,
        num_points=10000,
        sampling_method="poisson_disk",
        initial_sampling_factor=5,
        initial_point_cloud=None,
        use_triangle_normal=False,
    )

    # ===================== Log ================================================
    logger.success(f"Converted {cylinder_mesh} to a point cloud")
    logger.success(f"Results: {point_cloud}")
    logger.info(f"Point cloud has {len(point_cloud.positions)} points")

    # ===================== Visualization  (Optional) ===========================
    rr.init("convert_mesh_to_point_cloud_example", spawn=True)
    datatypes.visualize(cylinder_mesh, entity_path="/1-cylinder_mesh")
    datatypes.visualize(point_cloud, entity_path="/2-sampled_point_cloud")


if __name__ == "__main__":
    convert_mesh_to_point_cloud_example()
