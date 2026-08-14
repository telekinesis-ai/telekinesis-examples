"""
Demonstrates computing the convex hull mesh enclosing a point cloud.
"""

from loguru import logger
import rerun as rr

from telekinesis import vitreous, datatypes


def reconstruct_mesh_using_convex_hull_example():
    """
    Computes the convex hull mesh enclosing a point cloud.

    Creates the smallest convex shape that contains all points.
    """
    # ===================== Load Data ==========================================
    point_cloud_url = "https://assets.telekinesis.ai/examples/v1/point_clouds/beer_can_corrupted_normals.ply"
    point_cloud = datatypes.PointCloud.from_url(url=point_cloud_url, use_cache=True)

    # ===================== Run Skill ==========================================
    result_mesh = vitreous.reconstruct_mesh_using_convex_hull(
        joggle_inputs=False,
        point_cloud=point_cloud,
    )

    # ===================== Log ================================================
    logger.success(f"Reconstructed convex hull mesh from {point_cloud}")
    logger.success(f"Results: {result_mesh}")
    logger.info(
        f"Result mesh has {len(result_mesh.vertex_positions)} vertices and {len(result_mesh.triangle_indices)} triangles"
    )
    logger.info(f"Result mesh has vertex normals: {result_mesh.has_vertex_normals}")
    logger.info(f"Result mesh has vertex colors: {result_mesh.has_vertex_colors}")

    # ===================== Visualization  (Optional) ===========================
    rr.init("reconstruct_mesh_using_convex_hull_example", spawn=True)
    datatypes.visualize(point_cloud, entity_path="/1-input_point_cloud")
    datatypes.visualize(result_mesh, entity_path="/2-convex_hull_mesh")


if __name__ == "__main__":
    reconstruct_mesh_using_convex_hull_example()
