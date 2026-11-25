"""
Loads saved part-relative grasps, places them on a detected part pose and shows them in Rerun.

Runs offline: the example writes a small grasp file itself, so no hardware,
camera or robot is needed.

Supports all robots. No robot is needed.

Usage:
    python grasp_sampler.py
    python grasp_sampler.py --save-rrd grasp_sampler.rrd
"""

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import rerun as rr
from loguru import logger
from scipy.spatial.transform import Rotation

from telekinesis.synapse import grasps

# Length of the XYZ axes drawn for each grasp TCP and for the part, in meters.
GRASP_AXIS_LENGTH = 0.04
PART_AXIS_LENGTH = 0.08
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


def main(save_rrd: str | None) -> None:
    """Load two saved grasps and express them in the robot base frame."""

    #===================== Create the Grasp File ===============================
    # One six-value pose [x, y, z, rx, ry, rz] (meters, XYZ Euler degrees) and one
    # rigid 4x4 transform. Both are relative to the part frame.
    grasp_matrix = np.eye(4)
    grasp_matrix[:3, :3] = Rotation.from_euler("xyz", [0, 180, 0], degrees=True).as_matrix()
    grasp_matrix[:3, 3] = [0.0, 0.02, 0.01]
    grasp_file = {
        "part_name": "bracket_v2",
        "grasps": [
            [0.03, 0.0, 0.005, 180.0, 0.0, 0.0],
            grasp_matrix.tolist(),
        ],
    }
    path = Path(tempfile.mkdtemp()) / "bracket_v2.json"
    path.write_text(json.dumps(grasp_file), encoding="utf-8")

    #===================== Create the Sampler ==================================
    sampler = grasps.HardcodedGraspSampler("bracket_v2", path)
    if not sampler.has_grasps:
        logger.error("No usable grasps were loaded")
        return

    # ==================== Run Skill ============================================
    # Pose of the part frame in the robot base frame, e.g. from your detector.
    base_T_object = np.eye(4)
    base_T_object[:3, :3] = Rotation.from_euler("xyz", [0, 0, 90], degrees=True).as_matrix()
    base_T_object[:3, 3] = [0.4, 0.1, 0.05]

    base_T_tcps = sampler.execute(base_T_object=base_T_object)
    for index, base_T_tcp in enumerate(base_T_tcps):
        logger.success(f"Grasp {index} TCP position in base frame [m]: {np.round(base_T_tcp[:3, 3], 3)}")

    # ==================== Visualization ========================================
    # The part frame, every grasp TCP frame in file order, and a line from the
    # part origin to each grasp.
    recording = init_rerun("telekinesis_synapse_grasp_sampler", save_rrd)
    recording.log("/", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    log_frame(recording, "/grasp_sampler/part", base_T_object, "part", PART_AXIS_LENGTH)
    for index, base_T_tcp in enumerate(base_T_tcps):
        log_frame(recording, f"/grasp_sampler/grasps/{index}", base_T_tcp,
                  f"grasp {index}", GRASP_AXIS_LENGTH)
    recording.log(
        "/grasp_sampler/links",
        rr.LineStrips3D(
            [np.array([base_T_object[:3, 3], base_T_tcp[:3, 3]]) for base_T_tcp in base_T_tcps],
            colors=[[200, 200, 200]] * len(base_T_tcps),
            radii=0.0005,
        ),
        static=True,
    )
    recording.flush()
    if save_rrd:
        logger.info(f"Saved Rerun recording: {save_rrd}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Synapse grasp sampler example")
    parser.add_argument("--save-rrd", type=str, default=None,
                        help="Save the Rerun recording to this .rrd file instead of opening the viewer")
    args = parser.parse_args()

    main(save_rrd=args.save_rrd)
