"""
Demonstrates filtering point-cloud points using a mesh convex hull.
"""

from loguru import logger
import rerun as rr

from telekinesis import vitreous, datatypes


def filter_point_cloud_using_mesh_convex_hull_example():
    """
    Removes points that fall inside a mesh's convex hull.

    The mesh defines a convex filtering volume, including the space enclosed
    between its vertices even when the original mesh is not convex.
    """
    # ===================== Load Data ==========================================
    point_cloud_url = (
        "https://assets.telekinesis.ai/examples/v1/point_clouds/mesh_filter_example_pc.ply"
    )
    point_cloud = datatypes.PointCloud.from_url(url=point_cloud_url, use_cache=True)

    mesh_0 = datatypes.Mesh3D.from_url(url="https://assets.telekinesis.ai/examples/v1/meshes/mesh_0.ply", use_cache=True)
    mesh_1 = datatypes.Mesh3D.from_url(url="https://assets.telekinesis.ai/examples/v1/meshes/mesh_1.ply", use_cache=True)
    mesh_2 = datatypes.Mesh3D.from_url(url="https://assets.telekinesis.ai/examples/v1/meshes/mesh_2.ply", use_cache=True)
    mesh_3 = datatypes.Mesh3D.from_url(url="https://assets.telekinesis.ai/examples/v1/meshes/mesh_3.ply", use_cache=True)
    mesh_4 = datatypes.Mesh3D.from_url(url="https://assets.telekinesis.ai/examples/v1/meshes/mesh_4.ply", use_cache=True)
    mesh_5 = datatypes.Mesh3D.from_url(url="https://assets.telekinesis.ai/examples/v1/meshes/mesh_5.ply", use_cache=True)
    mesh_6 = datatypes.Mesh3D.from_url(url="https://assets.telekinesis.ai/examples/v1/meshes/mesh_6.ply", use_cache=True)
    mesh_7 = datatypes.Mesh3D.from_url(url="https://assets.telekinesis.ai/examples/v1/meshes/mesh_7.ply", use_cache=True)
    mesh_8 = datatypes.Mesh3D.from_url(url="https://assets.telekinesis.ai/examples/v1/meshes/mesh_8.ply", use_cache=True)

    meshes = datatypes.Mesh3DBatch(
        [mesh_0, mesh_1, mesh_2, mesh_3, mesh_4, mesh_5, mesh_6, mesh_7, mesh_8]
    )

    # ===================== Run Skill ==========================================
    filtered_point_cloud = vitreous.filter_point_cloud_using_mesh_convex_hull(
        point_cloud=point_cloud,
        meshes=meshes,
        mode="exclude",
        padding=0.0,
        tolerance=1e-9,
    )

    # ===================== Log ================================================
    removed_point_count = len(point_cloud.positions) - len(
        filtered_point_cloud.positions
    )
    logger.success(f"Filtered {point_cloud} using the convex hulls of {meshes}")
    logger.success(f"Results: {filtered_point_cloud}")
    logger.info(
        f"Kept {len(filtered_point_cloud.positions)} points and removed "
        f"{removed_point_count} points"
    )

    # ===================== Visualization  (Optional) ===========================
    rr.init("filter_point_cloud_using_mesh_convex_hull_example", spawn=True)
    datatypes.visualize(point_cloud, entity_path="/1-input_point_cloud")
    datatypes.visualize(meshes, entity_path="/2-filtering_meshes")
    datatypes.visualize(filtered_point_cloud, entity_path="/3-filtered_point_cloud")


if __name__ == "__main__":
    filter_point_cloud_using_mesh_convex_hull_example()
