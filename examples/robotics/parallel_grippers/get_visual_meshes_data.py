"""
Demonstrates how to read the per-link visual mesh data of a gripper URDF for
the Synapse SDK.

Supports all.

Usage:
    python get_visual_meshes_data.py
"""

import numpy as np
import rerun as rr
from loguru import logger

from telekinesis.synapse.tools.parallel_grippers import robotiq


def main() -> None:
    """Reads the visual mesh data of every link of a Robotiq gripper."""

    #===================== Create Gripper ======================================
    gripper = robotiq.Robotiq2F85()

    # ==================== Run Skill ===========================================
    meshes = gripper.get_visual_meshes_data()
    logger.info(f"Number of links: {len(meshes)}")

    # =================== Visualization (Optional) ==============================
    rr.init(f"telekinesis_synapse_{type(gripper).__name__}", spawn=True)

    # The mesh data is expressed in each link's own frame, so it needs the
    # matching link transform to be placed correctly. Without this every mesh
    # would be drawn at the world origin, stacked on top of each other.
    transforms = gripper.get_visual_mesh_transforms(base_transform=np.eye(4))

    # Log the vertex, triangle and color counts of each link's visual mesh, and
    # log the mesh geometry at its link transform
    for link_name, mesh in meshes.items():
        if mesh["vertices"] is None:
            logger.warning(f"{link_name}: no visual mesh")
            continue
        logger.success(f"{link_name}: "
                       f"vertices={mesh['vertices'].shape[0]}, "
                       f"triangles={mesh['triangles'].shape[0]}, "
                       f"vertex_colors={mesh['vertex_colors'] is not None}, "
                       f"mesh_origin={mesh['mesh_origin']}")

        kwargs = {"vertex_positions": mesh["vertices"],
                  "triangle_indices": mesh["triangles"],
                  "vertex_normals": mesh["vertex_normals"]}
        if mesh["vertex_colors"] is not None:
            kwargs["vertex_colors"] = mesh["vertex_colors"]
        else:
            kwargs["albedo_factor"] = mesh["color"] or [179, 179, 179]

        entity = f"/visual_meshes/{link_name}"
        rr.log(entity, rr.Mesh3D(**kwargs))

        transform = transforms.get(link_name)
        if transform is not None:
            rr.log(entity, rr.Transform3D(translation=transform[:3, 3],
                                          mat3x3=transform[:3, :3]))


if __name__ == "__main__":
    main()
