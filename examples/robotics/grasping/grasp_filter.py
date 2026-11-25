"""
Filters a batch of grasp poses with inverse kinematics and collision checks, and shows the result in Rerun.

Builds an Epson CX4 with a Robotiq Hand-E and a small box in a planning scene,
then filters five grasp poses and, for those that survive, their pregrasp poses.
Runs without a robot connection; the first run downloads the robot and gripper
models.

Supports Epson and virtual.

Usage:
    python grasp_filter.py
    python grasp_filter.py --save-rrd grasp_filter.rrd
"""

import argparse

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from loguru import logger

from telekinesis.synapse import grasps, planning_scene
from telekinesis.synapse.robots.manipulators import epson
from telekinesis.synapse.tools.parallel_grippers import robotiq

PRE_GRASP_OFFSET_M = 0.05


def init_rerun(app_id: str, save_rrd: str | None) -> "rr.RecordingStream":
    """Start a Rerun recording, live in the viewer or saved to an .rrd file."""
    rr.init(app_id, spawn=save_rrd is None)
    recording = rr.get_global_data_recording()
    if save_rrd:
        recording.save(save_rrd)
    return recording


def main(save_rrd: str | None) -> None:
    """Filter five grasp poses, log the outcome of each and show them in Rerun."""

    #===================== Create Robot and Gripper ============================
    robot = epson.EpsonCX4A601S()
    gripper = robotiq.RobotiqHandE()
    robot.attach_tool(gripper, mount_frame=robot.default_tcp)
    robot.add_tcp(name="grasp_tcp", transform=[0, 0, 0.15, 0, 0, 0], set_active=True)

    #===================== Create the Planning Scene ===========================
    # Create the scene after changing TCPs: it copies the robot model.
    scene = planning_scene.PlanningScene(robot.kinematic_model)

    # Put the gripper's collision model in the scene, at a 40 mm opening.
    gripper_configuration = gripper.get_model_configuration_at_opening_width(40, unit="mm")[:1]
    scene.attach_collision_object(
        "gripper",
        parent_frame=robot.default_tcp,
        touch_links=[robot.default_tcp, "j5_link", "j6_link"],
        parent_T_object=np.eye(4),
        urdf_path=gripper.urdf_path,
        package_dir=gripper.model_dir,
        configuration=gripper_configuration,
    )

    #===================== Create Candidate Grasps =============================
    # Five TCP poses in the robot base frame, shifted sideways around a reachable pose.
    seed_joints_deg = np.array([15, -35, 45, 10, 20, -10], dtype=float)
    base_T_center = robot.kinematic_model.forward_kinematics(
        np.deg2rad(seed_joints_deg), robot.active_tcp
    )
    grasp_poses = []
    for offset_y in np.linspace(-0.02, 0.02, 5):
        base_T_tcp = base_T_center.copy()
        base_T_tcp[:3, 3] += base_T_center[:3, 1] * offset_y
        grasp_poses.append(base_T_tcp)

    # A box next to the grasp area. Candidates that bring the robot or the
    # gripper into it are reported as COLLISION.
    base_T_box = base_T_center.copy()
    base_T_box[:3, 3] += base_T_center[:3, 1] * 0.09
    scene.world.add_box("obstacle", base_T_box, [0.04, 0.04, 0.04])

    # ==================== Run Skill ============================================
    results = grasps.GraspFilter().filter_grasps(
        grasp_candidates=grasp_poses,
        planning_scene=scene,
        ik_settings=robot.get_kinematics_settings(),
        ik_frame=robot.active_tcp,
        ik_solver_frame=robot.default_tcp,
        ik_seed_joint_positions_deg=seed_joints_deg,
        ik_joint_names=robot.joint_names,
        pre_grasp_offset_m=PRE_GRASP_OFFSET_M,
    )

    for result in results:
        logger.info(f"Grasp {result.candidate_index}: {result.status.name}, valid={result.is_valid}")
        if result.joint_positions_deg is not None:
            logger.info(f"  grasp joints [deg]: {np.round(result.joint_positions_deg, 2)}")
        if result.pregrasp_joint_positions_deg is not None:
            logger.info(f"  pregrasp joints [deg]: {np.round(result.pregrasp_joint_positions_deg, 2)}")
        if result.colliding_geometry_names:
            logger.warning(f"  colliding geometry: {result.colliding_geometry_names}")

    valid = [r.candidate_index for r in results if r.status == grasps.GraspFilterStatus.VALID]
    logger.success(f"Valid grasps: {valid}")

    # ==================== Visualization ========================================
    # The robot and scene, every candidate and pregrasp pose as an arrow colored by status,
    # and the solved configurations. Scrub the grasp_solution timeline to see the robot at
    # each solved candidate.
    recording = init_rerun("telekinesis_synapse_grasp_filter", save_rrd)
    recording.send_blueprint(
        rrb.Blueprint(
            rrb.Horizontal(
                rrb.Spatial3DView(origin="/"),
                rrb.TextLogView(origin="/grasp_filter/solution"),
                column_shares=[3, 1],
            ),
            rrb.TimePanel(timeline="grasp_solution", play_state="following"),
        )
    )
    recording.log("/", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    # The visualizer draws into the scene's recording, so start that first.
    scene.visualize_rerun(recording_stream=recording)
    visualizer = grasps.GraspFilterVisualizer()
    visualizer.visualize_grasps(grasp_poses=grasp_poses, results=results, planning_scene=scene)
    visualizer.visualize_pregrasp_poses(
        grasp_poses=grasp_poses,
        results=results,
        planning_scene=scene,
        pre_grasp_offset_m=PRE_GRASP_OFFSET_M,
    )
    visualizer.visualize_ik_solutions(results=results, planning_scene=scene, include_collisions=True)
    recording.flush()
    if save_rrd:
        logger.info(f"Saved Rerun recording: {save_rrd}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Synapse grasp filter example")
    parser.add_argument("--save-rrd", type=str, default=None,
                        help="Save the Rerun recording to this .rrd file instead of opening the viewer")
    args = parser.parse_args()

    main(save_rrd=args.save_rrd)
