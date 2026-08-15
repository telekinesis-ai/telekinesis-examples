"""
Example script to demonstrate usage of the Mesh3D datatype.

Shows:
  - constructing a Mesh3D from vertex positions, triangle indices,
    vertex normals, and vertex colors
  - accessing individual properties
  - visualizing the mesh using Rerun
  - loading a mesh from a URL / a local file path
  - saving a mesh to disk
  - round-tripping via `to_pyarrow` / `from_pyarrow`
"""

import numpy as np
from loguru import logger
import rerun as rr

from telekinesis import datatypes


def mesh3d_example():
    """
    Example function to demonstrate usage of the Mesh3D datatype.
        - Build a Mesh3D
        - Access vertex_positions / triangle_indices / vertex_normals / vertex_colors
        - Visualize the Mesh3D data using Rerun
        - Load a Mesh3D from a URL and from a local file
        - Save the Mesh3D to disk
        - Serialize and deserialize the Mesh3D
    """
    # Build a Mesh3D: a single-sided tetrahedron with per-vertex normals and colors.
    vertex_positions = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32
    )
    triangle_indices = np.array(
        [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=np.int32
    )
    vertex_normals = np.array(
        [[0, 0, -1], [0, -1, 0], [-1, 0, 0], [1, 1, 1]], dtype=np.float32
    )
    vertex_colors = np.array(
        [[255, 0, 0, 255], [0, 255, 0, 255], [0, 0, 255, 255], [255, 255, 0, 255]],
        dtype=np.uint8,
    )

    my_mesh = datatypes.Mesh3D(
        vertex_positions=vertex_positions,
        triangle_indices=triangle_indices,
        vertex_normals=vertex_normals,
        vertex_colors=vertex_colors,
    )
    logger.info(f"Original Mesh3D: {my_mesh}")

    # Access underlying vertex_positions / triangle_indices / vertex_normals / vertex_colors
    my_mesh_vertex_positions = my_mesh.vertex_positions
    my_mesh_triangle_indices = my_mesh.triangle_indices
    my_mesh_vertex_normals = my_mesh.vertex_normals
    my_mesh_vertex_colors = my_mesh.vertex_colors
    logger.info(f"Underlying Mesh3D vertex_positions: {my_mesh_vertex_positions}")
    logger.info(f"Underlying Mesh3D triangle_indices: {my_mesh_triangle_indices}")
    logger.info(f"Underlying Mesh3D vertex_normals: {my_mesh_vertex_normals}")
    logger.info(f"Underlying Mesh3D vertex_colors (packed RGBA uint32): {my_mesh_vertex_colors}")
    logger.info(f"Number of vertices: {len(my_mesh)}")
    logger.info(f"Has vertex normals: {my_mesh.has_vertex_normals}")
    logger.info(f"Has vertex colors: {my_mesh.has_vertex_colors}")

    # Mesh3D has no `datatypes.visualize()` handler yet, so it is logged directly with Rerun.
    rr.init("mesh3d_example", spawn=True)
    rr.log(
        "/Mesh3D/my_mesh",
        rr.Mesh3D(
            vertex_positions=my_mesh.vertex_positions,
            triangle_indices=my_mesh.triangle_indices,
            vertex_normals=my_mesh.vertex_normals,
            vertex_colors=my_mesh.vertex_colors,
        ),
    )

    # Load a mesh from a URL. By default, it will be cached in the user cache
    # directory for future runs.
    mesh_url = "https://assets.telekinesis.ai/examples/v1/meshes/gear_box.glb"
    my_mesh_from_url = datatypes.Mesh3D.from_url(url=mesh_url, use_cache=True)
    logger.info(f"My new Mesh3D from URL: {my_mesh_from_url}")

    # Save to disk, then load it back from the local path.
    my_mesh.save_to_path("results/my_mesh_saved.ply")
    logger.info("Saved Mesh3D to disk as .ply file.")

    my_mesh_from_path = datatypes.Mesh3D.from_path("results/my_mesh_saved.ply")
    logger.info(f"My new Mesh3D from .ply: {my_mesh_from_path}")

    # Serialize to PyArrow and back.
    serialized_mesh = datatypes.serialize(my_mesh)
    deserialized_mesh = datatypes.deserialize(serialized_mesh)["param_0"]
    logger.info(f"Deserialized Mesh3D: {deserialized_mesh}")
    # Mesh3D doesn't implement `__eq__`, so compare the underlying arrays directly.
    logger.info(
        "Deserialized Mesh3D matches original: "
        f"{np.array_equal(deserialized_mesh.vertex_positions, my_mesh.vertex_positions) and np.array_equal(deserialized_mesh.triangle_indices, my_mesh.triangle_indices) and np.array_equal(deserialized_mesh.vertex_normals, my_mesh.vertex_normals) and np.array_equal(deserialized_mesh.vertex_colors, my_mesh.vertex_colors)}"
    )


if __name__ == "__main__":
    mesh3d_example()
