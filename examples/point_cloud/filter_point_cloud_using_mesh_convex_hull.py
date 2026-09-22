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
        "https://assets.telekinesis.ai/examples/v1/point_clouds/plastic_2_raw.ply"
    )
    mesh_url = "https://assets.telekinesis.ai/examples/v1/meshes/beer_can.glb"
    point_cloud = datatypes.PointCloud.from_url(url=point_cloud_url, use_cache=True)
    mesh = datatypes.Mesh3D.from_url(url=mesh_url, use_cache=True)

    # ===================== Run Skill ==========================================
    filtered_point_cloud = vitreous.filter_point_cloud_using_mesh_convex_hull(
        point_cloud=point_cloud,
        meshes=mesh,
        mode="exclude",
        padding=0.0,
        tolerance=1e-9,
    )

    # ===================== Log ================================================
    removed_point_count = len(point_cloud.positions) - len(
        filtered_point_cloud.positions
    )
    logger.success(f"Filtered {point_cloud} using the convex hull of {mesh}")
    logger.success(f"Results: {filtered_point_cloud}")
    logger.info(
        f"Kept {len(filtered_point_cloud.positions)} points and removed "
        f"{removed_point_count} points"
    )

    # ===================== Visualization  (Optional) ===========================
    rr.init("filter_point_cloud_using_mesh_convex_hull_example", spawn=True)
    datatypes.visualize(point_cloud, entity_path="/1-input_point_cloud")
    datatypes.visualize(mesh, entity_path="/2-filtering_mesh")
    datatypes.visualize(filtered_point_cloud, entity_path="/3-filtered_point_cloud")


if __name__ == "__main__":
    filter_point_cloud_using_mesh_convex_hull_example()
