"""
Queries a UR10e's kinematic model: forward kinematics, frames, joint limits and self-collision,
visualized in Rerun.

Supports Universal Robots (UR), Epson, and virtual. Runs offline, no hardware needed.

Usage:
    python kinematic_model.py
    python kinematic_model.py --save-rrd kinematic_model.rrd
"""

import argparse

import numpy as np
import rerun as rr
from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots

Q_DEG = [0.0, -90.0, 90.0, 0.0, 90.0, 0.0]


def init_rerun(app_id: str, save_rrd: str | None) -> "rr.RecordingStream":
    """Start a Rerun recording, live in the viewer or saved to an .rrd file."""
    rr.init(app_id, spawn=save_rrd is None)
    recording = rr.get_global_data_recording()
    if save_rrd:
        recording.save(save_rrd)
    return recording


def log_frame(recording: "rr.RecordingStream", path: str, pose: np.ndarray) -> None:
    """Draw an XYZ axis triad at a 4 x 4 frame pose, in meters."""
    recording.log(path, rr.Transform3D(translation=pose[:3, 3], mat3x3=pose[:3, :3]), static=True)
    recording.log(
        f"{path}/axes",
        rr.Arrows3D(
            origins=np.zeros((3, 3)),
            vectors=np.eye(3) * 0.03,
            colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
        ),
        static=True,
    )


def main(save_rrd: str | None) -> None:
    """Read the model's structure and evaluate it at one joint configuration."""

    #===================== Create Robot ==========================================
    robot = universal_robots.UniversalRobotsUR10E(name='UR10e')
    model = robot.kinematic_model
    q_rad = np.deg2rad(Q_DEG)

    # ==================== Visualization ========================================
    # Move the robot to the configuration, then draw it in the same recording.
    recording = init_rerun("telekinesis_synapse_kinematic_model", save_rrd)
    recording.log("/", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    robot.set_joint_positions(Q_DEG)
    robot.visualize_rerun(recording_stream=recording, live=False)

    # ==================== Run Skill ============================================
    # Structure. These are methods on the model, so call them.
    logger.info(f"nq={model.nq()}, ndof={model.ndof()}")
    logger.info(f"Joint names: {model.joint_names()}")
    logger.info(f"Frames: {len(model.frame_names())}")

    # Forward kinematics: a 4x4 pose of one frame, or of every frame in one call.
    world_T_tcp = model.forward_kinematics(q_rad, robot.default_tcp)
    logger.info(f"TCP position (m): {np.round(world_T_tcp[:3, 3], 4).tolist()}")
    transforms = model.all_frame_transforms(q_rad)
    logger.info(f"Frame transforms returned: {len(transforms)}")

    # Every frame of the model at this configuration, as an axis triad.
    for frame_name, world_T_frame in transforms.items():
        log_frame(recording, f"/kinematic_model/frames/{frame_name}", np.asarray(world_T_frame))

    # Joint limits, in radians.
    lower_deg = np.rad2deg(model.lower_position_limits())
    upper_deg = np.rad2deg(model.upper_position_limits())
    logger.info(f"Lower limits (deg): {np.round(lower_deg, 1).tolist()}")
    logger.info(f"Upper limits (deg): {np.round(upper_deg, 1).tolist()}")
    logger.info(f"Strictly within limits: {model.in_joint_limits(q_rad)}")

    # Self-collision and the neutral configuration.
    logger.info(f"Self-collision: {model.has_self_collision(q_rad)}")
    logger.info(f"Neutral configuration (rad): {model.neutral_configuration().tolist()}")

    # Add a frame at a fixed offset from the default TCP. robot.add_tcp adds the frame to this
    # model and records the TCP, so use it rather than calling model.add_frame yourself. The
    # robot is redrawn with the new frame.
    robot.add_tcp(name="probe", transform=[0.0, 0.0, 0.1, 0.0, 0.0, 0.0], set_active=False)
    logger.info(f"'probe' frame exists: {'probe' in model.frame_names()}")

    recording.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kinematic model Synapse example")
    parser.add_argument("--save-rrd", type=str, default=None,
                        help="Save the Rerun recording to this .rrd file instead of opening the viewer")
    args = parser.parse_args()

    main(save_rrd=args.save_rrd)
