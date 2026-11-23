"""
Solves inverse kinematics with the native solver classes, without a robot connection,
visualized in Rerun.

Builds a reachable target from a known joint configuration, solves it with TRAC-IK,
then solves the same target again with a validity callback and with profiling.

Supports every manipulator. Universal Robots (UR10e) is used here for illustration.

Usage:
    python kinematic_solvers_example.py
    python kinematic_solvers_example.py --save-rrd kinematic_solvers.rrd

Note:
    This file is not named kinematic_solvers.py on purpose: Synapse imports the native
    ``kinematic_solvers`` module by that name, and a script with the same name next to it
    would shadow the module.
"""

import argparse

import kinematic_solvers
import numpy as np
import rerun as rr
from loguru import logger

from telekinesis.synapse import planning_scene
from telekinesis.synapse.robots.manipulators import universal_robots


def init_rerun(app_id: str, save_rrd: str | None) -> "rr.RecordingStream":
    """Start a Rerun recording, live in the viewer or saved to an .rrd file."""
    rr.init(app_id, spawn=save_rrd is None)
    recording = rr.get_global_data_recording()
    if save_rrd:
        recording.save(save_rrd)
    return recording


def log_tcp_frame(recording: "rr.RecordingStream", path: str, pose: np.ndarray, static: bool) -> None:
    """Draw an XYZ axis triad at a 4 x 4 TCP pose, in meters."""
    recording.log(path, rr.Transform3D(translation=pose[:3, 3], mat3x3=pose[:3, :3]), static=static)
    recording.log(
        f"{path}/axes",
        rr.Arrows3D(
            origins=np.zeros((3, 3)),
            vectors=np.eye(3) * 0.05,
            colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
        ),
        static=static,
    )


def main(save_rrd: str | None) -> None:
    """Solve one pose with TRAC-IK, a validity callback and profiling."""

    #===================== Create Robot ==========================================
    # No connection is needed: the solver only uses the kinematic model.
    robot = universal_robots.UniversalRobotsUR10E(name="UR10e")
    model = robot.kinematic_model
    frame = robot.default_tcp

    #===================== Build a Reachable Target ==============================
    q_known_rad = np.deg2rad([0.0, -95.7, -90.0, 0.0, 90.0, 0.0])
    target_pose = model.forward_kinematics(q_known_rad, frame)

    # ==================== Visualization ========================================
    # A planning scene draws the robot at each configuration below. The target frame stays
    # fixed, and the solution frame lands on it when the solve succeeds.
    scene = planning_scene.PlanningScene(kinematic_model=model)
    recording = init_rerun("telekinesis_synapse_kinematic_solvers", save_rrd)
    recording.log("/", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    scene.visualize_rerun(recording_stream=recording)
    log_tcp_frame(recording, "/ik/target_tcp", target_pose, static=True)

    def show(step: int, q_rad: np.ndarray) -> None:
        """Draw the robot at q_rad and the TCP frame that q_rad reaches."""
        recording.set_time("ik_step", sequence=step)
        scene.visualize_configuration_rerun(np.rad2deg(q_rad), [])
        log_tcp_frame(recording, "/ik/solution_tcp", model.forward_kinematics(q_rad, frame), static=False)

    show(0, q_known_rad)

    #===================== Create the Solver =====================================
    settings = kinematic_solvers.KinematicsSettings("tracik", timeout_s=0.1)
    solver = kinematic_solvers.create_configured_solver(model, frame, settings)
    logger.info(f"Solver: {solver}")

    # ==================== Run Skill ============================================
    # One entry per target pose: a joint vector in radians, or None.
    solutions = solver.solve([target_pose], q_inits=[q_known_rad])
    if solutions[0] is None:
        logger.error("No IK solution found")
        return
    logger.success(f"IK solution [deg]: {np.rad2deg(solutions[0]).round(2).tolist()}")
    show(1, solutions[0])

    # Search from one seed and accept only solutions your own check allows.
    # is_valid receives model-ordered joints in radians.
    def within_wrist_limit(q_rad: np.ndarray) -> bool:
        return abs(float(q_rad[5])) < np.deg2rad(90.0)

    accepted = solver.solve_with_validity_callback(
        target_pose, q_known_rad, within_wrist_limit
    )
    logger.info(f"Validity-checked solution found: {accepted is not None}")
    if accepted is not None:
        show(2, accepted)

    # Same solve, with timing and error details per pose.
    result = solver.solve_profiled([target_pose], q_inits=[q_known_rad])[0]
    logger.info(
        f"converged={result.converged}, "
        f"linear error={result.linear_error_norm_meters:.2e} m, "
        f"angular error={result.angular_error_norm_rad:.2e} rad, "
        f"seeds tried={result.num_seeds_tried}"
    )
    recording.log(
        "/ik/profile",
        rr.TextLog(
            f"converged={result.converged}; linear error={result.linear_error_norm_meters:.2e} m; "
            f"angular error={result.angular_error_norm_rad:.2e} rad; seeds tried={result.num_seeds_tried}"
        ),
    )

    recording.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kinematic solvers Synapse example")
    parser.add_argument("--save-rrd", type=str, default=None,
                        help="Save the Rerun recording to this .rrd file instead of opening the viewer")
    args = parser.parse_args()

    main(save_rrd=args.save_rrd)
