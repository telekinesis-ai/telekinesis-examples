"""
Gets the visual meshes of a parallel gripper at different openings, and shows them side by side in Rerun.

Supports all parallel grippers with a URDF model.

Usage:
    python get_visual_meshes.py
    python get_visual_meshes.py --save-rrd get_visual_meshes.rrd
"""

import argparse

import rerun as rr
from loguru import logger

from telekinesis.synapse.tools.parallel_grippers import onrobot

# Distance between the gripper copies drawn along the y-axis, in meters.
SPACING_M = 0.2


def init_rerun(app_id: str, save_rrd: str | None) -> "rr.RecordingStream":
    """Start a Rerun recording, live in the viewer or saved to an .rrd file."""
    rr.init(app_id, spawn=save_rrd is None)
    recording = rr.get_global_data_recording()
    if save_rrd:
        recording.save(save_rrd)
    return recording


def log_meshes(path: str, meshes, y_offset: float, color: list[int]) -> None:
    """Draw a Mesh3DBatch shifted along the y-axis, one Rerun mesh per link."""
    rr.log(path, rr.Transform3D(translation=[0.0, y_offset, 0.0]))
    for index, (vertices, triangles) in enumerate(zip(meshes.vertex_positions,
                                                      meshes.triangle_indices)):
        rr.log(f"{path}/link_{index}",
               rr.Mesh3D(vertex_positions=vertices,
                         triangle_indices=triangles,
                         albedo_factor=color))


def main(save_rrd: str | None) -> None:
    """Queries an OnRobot gripper's meshes at several openings."""

    #===================== Start Rerun ==========================================
    init_rerun("telekinesis_synapse_get_visual_meshes", save_rrd)

    #===================== Create Gripper ======================================
    gripper = onrobot.OnRobotRG6()

    # ==================== Run Skill ============================================
    # Joint configuration of the model, in URDF units.
    q = gripper.get_model_configuration()
    meshes = gripper.get_visual_meshes_at_configuration(q)
    logger.success(f"At the model configuration: {len(meshes.vertex_positions)} meshes")

    # Half of the opening joint travel.
    half_open = gripper.get_visual_meshes_at_joint_travel_fraction(0.5)
    logger.info(f"At half travel: {len(half_open.vertex_positions)} meshes")

    # A physical jaw opening, up to the physical maximum of the model.
    open_25mm = gripper.get_visual_meshes_at_opening_width(25.0, unit="mm")
    logger.info(f"At a 25 mm opening: {len(open_25mm.vertex_positions)} meshes")

    # ==================== Visualization (Optional) =============================
    # The three queries are drawn side by side in the tool base frame:
    # the model configuration, half of the travel, and a 25 mm opening.
    log_meshes("/meshes/model_configuration", meshes, 0.0, [179, 179, 179])
    log_meshes("/meshes/half_travel", half_open, SPACING_M, [100, 160, 220])
    log_meshes("/meshes/opening_25mm", open_25mm, 2 * SPACING_M, [230, 160, 80])
    rr.log("/meshes/legend",
           rr.TextLog("Left to right: model configuration, half of the travel, 25 mm opening"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gripper get visual meshes Synapse example")
    parser.add_argument("--save-rrd", type=str, default=None,
                        help="Save the Rerun recording to this .rrd file instead of opening the viewer")
    args = parser.parse_args()

    main(save_rrd=args.save_rrd)
