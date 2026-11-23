"""
Generates the equivalent orientations of a parallel-jaw grasp, ranks them and shows them in Rerun.

Runs offline with numpy only: no robot, scene or hardware is needed.

Supports all robots. No robot is needed.

Usage:
    python symmetric_grasp.py
    python symmetric_grasp.py --save-rrd symmetric_grasp.rrd
"""

import argparse

import numpy as np
import rerun as rr
from loguru import logger
from scipy.spatial.transform import Rotation

from telekinesis.synapse import grasps

# The TCP-local axis the gripper is symmetric about (its approach axis), and the base-frame
# direction the TCP's Y axis should point toward.
SYMMETRY_AXIS = (0.0, 0.0, 1.0)
TARGET_DIRECTION = (0.0, 1.0, 0.0)

# Length of the XYZ axes drawn for each grasp, in meters, and the sideways distance between
# the drawn twins. The twins share one position; they are spread out only so they stay visible.
AXIS_LENGTH = 0.05
DISPLAY_SPACING = 0.12
AXIS_COLORS = [[255, 0, 0], [0, 255, 0], [0, 0, 255]]


def init_rerun(app_id: str, save_rrd: str | None) -> "rr.RecordingStream":
    """Start a Rerun recording, live in the viewer or saved to an .rrd file."""
    rr.init(app_id, spawn=save_rrd is None)
    recording = rr.get_global_data_recording()
    if save_rrd:
        recording.save(save_rrd)
    return recording


def log_frame(recording: "rr.RecordingStream", path: str, pose: np.ndarray,
              label: str, axis_length: float) -> None:
    """Log a 4x4 pose as labeled XYZ axes (red, green, blue) at an entity path."""
    recording.log(path, rr.Transform3D(translation=pose[:3, 3], mat3x3=pose[:3, :3]), static=True)
    recording.log(
        f"{path}/axes",
        rr.Arrows3D(origins=np.zeros((3, 3)), vectors=np.eye(3) * axis_length, colors=AXIS_COLORS),
        static=True,
    )
    recording.log(f"{path}/label", rr.Points3D(positions=[[0.0, 0.0, 0.0]], labels=[label]), static=True)


def contains_pose(poses: list[np.ndarray], pose: np.ndarray) -> int | None:
    """Return the index of ``pose`` in ``poses``, or None when it is not there."""
    for index, candidate in enumerate(poses):
        if np.allclose(candidate, pose):
            return index
    return None


def main(save_rrd: str | None) -> None:
    """Flip a grasp about the TCP approach axis and prefer the twin whose Y axis points to base +Y."""

    #===================== Create the Grasp Pose ===============================
    # TCP pointing down (rx = 180 deg), turned 30 deg about the vertical axis.
    base_T_tcp = np.eye(4)
    base_T_tcp[:3, :3] = Rotation.from_euler("xyz", [180, 0, 30], degrees=True).as_matrix()
    base_T_tcp[:3, 3] = [0.4, 0.1, 0.2]

    #===================== Create the Symmetry =================================
    # A parallel-jaw gripper is self-similar every 180 degrees about its approach (Z) axis.
    symmetry = grasps.SymmetricGrasp(symmetry_axis=SYMMETRY_AXIS, fold=2)

    # ==================== Run Skill ============================================
    orientations = symmetry.equivalent_orientations(base_T_tcp)
    for index, pose in enumerate(orientations):
        logger.info(f"Equivalent {index}: TCP Y axis in base frame = {np.round(pose[:3, 1], 3)}")

    ranked = symmetry.rank_by_desired_orientation(
        orientations,
        pick_point_axis=1,
        target_direction=TARGET_DIRECTION,
    )
    best = ranked[0]
    logger.success(f"Preferred TCP Y axis in base frame: {np.round(best[:3, 1], 3)}")
    logger.success(f"Candidates kept: {len(ranked)}")

    # A tolerance drops candidates that point too far from the target instead of only reordering them.
    within_60 = symmetry.rank_by_desired_orientation(
        orientations,
        pick_point_axis=1,
        target_direction=TARGET_DIRECTION,
        max_deviation_deg=60.0,
    )
    logger.success(f"Candidates within 60 deg: {len(within_60)}")

    # ==================== Visualization ========================================
    # The symmetry axis (grey line), the target direction (yellow arrow), and every equivalent
    # orientation as XYZ axes, labeled with its rank. Rank 0 is the preferred twin.
    recording = init_rerun("telekinesis_synapse_symmetric_grasp", save_rrd)
    recording.log("/", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    position = base_T_tcp[:3, 3]
    symmetry_axis_in_base = base_T_tcp[:3, :3] @ np.asarray(SYMMETRY_AXIS)
    recording.log(
        "/symmetric_grasp/symmetry_axis",
        rr.LineStrips3D(
            [np.array([position - symmetry_axis_in_base * 0.12, position + symmetry_axis_in_base * 0.12])],
            colors=[[200, 200, 200]],
            radii=0.001,
        ),
        static=True,
    )
    recording.log(
        "/symmetric_grasp/target_direction",
        rr.Arrows3D(
            origins=[position],
            vectors=[np.asarray(TARGET_DIRECTION) * 0.1],
            colors=[[255, 200, 0]],
            labels=["target direction"],
        ),
        static=True,
    )
    for index, pose in enumerate(orientations):
        rank = contains_pose(ranked, pose)
        within = contains_pose(within_60, pose) is not None
        label = f"equivalent {index}: rank {rank}" if rank is not None else f"equivalent {index}: dropped"
        label += ", within 60 deg" if within else ", outside 60 deg"
        shown = pose.copy()
        shown[0, 3] += index * DISPLAY_SPACING
        log_frame(recording, f"/symmetric_grasp/equivalents/{index}", shown, label, AXIS_LENGTH)
    recording.flush()
    if save_rrd:
        logger.info(f"Saved Rerun recording: {save_rrd}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Synapse symmetric grasp example")
    parser.add_argument("--save-rrd", type=str, default=None,
                        help="Save the Rerun recording to this .rrd file instead of opening the viewer")
    args = parser.parse_args()

    main(save_rrd=args.save_rrd)
