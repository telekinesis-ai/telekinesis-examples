"""
Reads the geometry of a Piab suction gripper without connecting to hardware, and shows it in Rerun.

The gripper's URDF is downloaded when the object is created, so the first run needs
network access. Nothing is connected or moved.

Supports Piab. The Isaac Sim SuctionGripper and SpotWeldingGun are USD-modelled and
raise NotImplementedError for these calls.

Usage:
    python get_geometry.py
    python get_geometry.py --save-rrd get_geometry.rrd
"""

import argparse

import numpy as np
import rerun as rr
from loguru import logger

from telekinesis.synapse.tools.suction_grippers import piab

# Length of the axis arrows drawn at each link, in meters.
AXIS_LENGTH = 0.02

# Distance between the gripper and the copy of its posed meshes along the y-axis, in meters.
SPACING_M = 0.2


def init_rerun(app_id: str, save_rrd: str | None) -> "rr.RecordingStream":
    """Start a Rerun recording, live in the viewer or saved to an .rrd file."""
    rr.init(app_id, spawn=save_rrd is None)
    recording = rr.get_global_data_recording()
    if save_rrd:
        recording.save(save_rrd)
    return recording


def main(save_rrd: str | None) -> None:
    """Log the link transforms, mesh poses and mesh data of a Piab gripper."""

    #===================== Start Rerun ==========================================
    recording = init_rerun("telekinesis_synapse_get_geometry", save_rrd)

    #===================== Create Gripper ======================================
    gripper = piab.PiabPiCobotElectric()

    # Place the tool base 0.3 m above the world origin.
    world_T_tool = np.eye(4)
    world_T_tool[2, 3] = 0.3

    try:
        # ==================== Run Skill ============================================
        configuration = gripper.get_model_configuration()
        logger.info(f"Model configuration: {configuration}")

        # Link poses in the tool base frame
        link_transforms = gripper.get_link_transforms()
        logger.success(f"Links: {list(link_transforms)}")

        # Visual meshes posed in the tool base frame for a given configuration.
        # This does not connect to hardware or change the gripper's state.
        meshes = gripper.get_visual_meshes_at_configuration(configuration)
        logger.success(f"Visual meshes: {type(meshes).__name__}")

        # World-frame pose of each link's mesh, for the tool base placed above
        mesh_poses = gripper.get_visual_mesh_transforms(world_T_tool)
        logger.success(f"Mesh poses for links: {list(mesh_poses)}")

        # Raw per-link mesh arrays, for a renderer of your own
        mesh_data = gripper.get_visual_meshes_data()
        for link_name, link_data in mesh_data.items():
            logger.info(f"{link_name}: {sorted(link_data)}")
    except NotImplementedError as e:
        logger.error(f"This tool has no URDF geometry: {e}")
        return

    # ==================== Visualization (Optional) =============================
    # The gripper is drawn at world_T_tool, which uses the link poses and mesh poses
    # above. Each link gets a colored frame, and the posed meshes are drawn again
    # beside the gripper, in the tool base frame.
    gripper.visualize_rerun(tcp_transform=world_T_tool, recording_stream=recording)

    for name, base_T_link in link_transforms.items():
        world_T_link = world_T_tool @ base_T_link
        origin = world_T_link[:3, 3]
        rr.log(f"/frames/{name}",
               rr.Arrows3D(origins=[origin, origin, origin],
                           vectors=[world_T_link[:3, axis] * AXIS_LENGTH for axis in range(3)],
                           colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]]))

    rr.log("/posed_meshes", rr.Transform3D(translation=[0.0, SPACING_M, world_T_tool[2, 3]]))
    for index, (vertices, triangles) in enumerate(zip(meshes.vertex_positions,
                                                      meshes.triangle_indices)):
        rr.log(f"/posed_meshes/link_{index}",
               rr.Mesh3D(vertex_positions=vertices,
                         triangle_indices=triangles,
                         albedo_factor=[179, 179, 179]))
    rr.log("/posed_meshes/legend",
           rr.TextLog("Left: the gripper at world_T_tool. Right: the posed meshes in the tool base frame."))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Suction gripper get geometry Synapse example")
    parser.add_argument("--save-rrd", type=str, default=None,
                        help="Save the Rerun recording to this .rrd file instead of opening the viewer")
    args = parser.parse_args()

    main(save_rrd=args.save_rrd)
