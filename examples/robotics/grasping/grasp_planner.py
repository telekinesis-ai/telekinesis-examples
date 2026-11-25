"""
Plans the approach, gripper closing, lift and retreat of filtered grasps, and replays the paths in Rerun.

Builds an Epson CX4 with a Robotiq Hand-E in a planning scene, filters five
grasp poses, then plans motions around them with each GraspPlanner method.
Runs without a robot connection; the first run downloads the robot and gripper
models.

Supports Epson and virtual.

Usage:
    python grasp_planner.py
    python grasp_planner.py --save-rrd grasp_planner.rrd
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
    """Filter five grasps, plan the motions around the ones that survive and replay them."""

    #===================== Create Robot and Gripper ============================
    robot = epson.EpsonCX4A601S()
    gripper = robotiq.RobotiqHandE()
    robot.attach_tool(gripper, mount_frame=robot.default_tcp)
    robot.add_tcp(name="grasp_tcp", transform=[0, 0, 0.15, 0, 0, 0], set_active=True)

    #===================== Create the Planning Scene ===========================
    # The scene starts with the gripper open, so checks use its open shape.
    scene = planning_scene.PlanningScene(robot.kinematic_model)
    open_configuration = gripper.get_model_configuration_at_opening_width(40, unit="mm")[:1]
    closed_configuration = gripper.get_model_configuration_at_opening_width(0, unit="mm")[:1]
    scene.attach_collision_object(
        "gripper",
        parent_frame=robot.default_tcp,
        touch_links=[robot.default_tcp, "j5_link", "j6_link"],
        parent_T_object=np.eye(4),
        urdf_path=gripper.urdf_path,
        package_dir=gripper.model_dir,
        configuration=open_configuration,
    )

    #===================== Create Candidate Grasps =============================
    # A candidate is a dictionary with a 4x4 "base_T_pick" in the robot base frame.
    seed_joints_deg = np.array([15, -35, 45, 10, 20, -10], dtype=float)
    base_T_center = robot.kinematic_model.forward_kinematics(
        np.deg2rad(seed_joints_deg), robot.active_tcp
    )
    candidates = []
    for offset_y in np.linspace(-0.02, 0.02, 5):
        base_T_pick = base_T_center.copy()
        base_T_pick[:3, 3] += base_T_center[:3, 1] * offset_y
        candidates.append({"base_T_pick": base_T_pick})

    # The planner needs the filter's joint solutions, so filter with a pregrasp offset.
    filter_results = grasps.GraspFilter().filter_grasps(
        grasp_candidates=[candidate["base_T_pick"] for candidate in candidates],
        planning_scene=scene,
        ik_settings=robot.get_kinematics_settings(),
        ik_frame=robot.active_tcp,
        ik_solver_frame=robot.default_tcp,
        ik_seed_joint_positions_deg=seed_joints_deg,
        ik_joint_names=robot.joint_names,
        pre_grasp_offset_m=PRE_GRASP_OFFSET_M,
    )

    #===================== Create the Grasp Data ===============================
    # Settings shared by every candidate.
    data = grasps.GraspData()
    data.tcp_frame = robot.active_tcp
    data.ik_solver_frame = robot.default_tcp
    data.attachment_frame = robot.default_tcp
    data.gripper_object_name = "gripper"
    data.ee_link_names = [robot.default_tcp]
    data.touch_links = [robot.default_tcp]
    data.open_gripper_configuration = open_configuration
    data.closed_gripper_configuration = closed_configuration
    data.approach_direction_tcp = np.array([0.0, 0.0, 1.0])
    data.approach_distance_m = PRE_GRASP_OFFSET_M
    data.lift_distance_m = 0.05
    data.retreat_direction_tcp = np.array([0.0, 0.0, -1.0])
    data.retreat_distance_m = 0.05
    data.ik_settings = robot.get_kinematics_settings()

    # ==================== Run Skill ============================================
    planner = grasps.GraspPlanner()

    # 1. Approach only: pregrasp to grasp.
    approaches = planner.plan_all_approaches(
        grasp_candidates=candidates,
        grasp_filter_results=filter_results,
        planning_scene=scene,
        grasp_data=data,
    )
    for plan in approaches:
        logger.info(
            f"Approach {plan.candidate_index}: accepted={plan.accepted}, "
            f"fraction={plan.approach_fraction:.2f}, reason={plan.rejection_reason!r}"
        )

    # 2. Approach, then close the gripper at the grasp. This scene has no target
    # occupancy in its OctoMap, so it is passed for both scenes.
    closed = planner.plan_all_approach_and_close(
        grasp_candidates=candidates,
        grasp_filter_results=filter_results,
        planning_scene=scene,
        target_excluded_scene=scene,
        grasp_data=data,
    )
    for plan in closed:
        logger.info(
            f"Approach and close {plan.candidate_index}: accepted={plan.accepted}, "
            f"closing fraction={plan.gripper_closing_fraction:.2f}"
        )

    # 3. Gripper closing alone, with the arm fixed at a known grasp configuration.
    for result in filter_results:
        if result.is_valid:
            close_plan = planner.plan_close(
                target_excluded_scene=scene,
                grasp_data=data,
                grasp_joint_positions_deg=result.joint_positions_deg,
            )
            logger.info(
                f"Close {result.candidate_index}: accepted={close_plan.accepted}, "
                f"colliding={close_plan.closing_colliding_geometry_names}"
            )
            break

    # 4. The full sequence: approach, lift and retreat.
    plans = planner.plan_all_approach_lift_retreat(
        grasp_candidates=candidates,
        grasp_filter_results=filter_results,
        planning_scene=scene,
        grasp_data=data,
    )
    for plan in plans:
        logger.info(
            f"Candidate {plan.candidate_index}: filter={filter_results[plan.candidate_index].status.name}, "
            f"accepted={plan.accepted}, fractions=({plan.approach_fraction:.2f}, "
            f"{plan.lift_fraction:.2f}, {plan.retreat_fraction:.2f}), reason={plan.rejection_reason!r}"
        )
        if plan.accepted:
            logger.success(
                f"  lift path: {len(plan.lift_joint_trajectory_deg)} states, "
                f"last joints [deg]: {np.round(plan.lift_joint_trajectory_deg[-1], 2)}"
            )

    # ==================== Visualization ========================================
    # Every candidate as a point (green = accepted, red = rejected), the filter's candidate
    # and pregrasp arrows, then a replay of the approach, lift and retreat paths of the full
    # sequence. The gripper is shown open on the approach and closed on the lift and retreat.
    # The replay only plays planned states. It does not move any robot.
    recording = init_rerun("telekinesis_synapse_grasp_planner", save_rrd)
    recording.send_blueprint(
        rrb.Blueprint(
            rrb.Horizontal(
                rrb.Spatial3DView(origin="/"),
                rrb.TextLogView(origin="/grasp_planner"),
                column_shares=[3, 1],
            ),
            rrb.TimePanel(timeline="grasp_planning_step", play_state="following"),
        )
    )
    recording.log("/", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    scene.visualize_rerun(recording_stream=recording, retain_collision_checks=True)

    poses = [candidate["base_T_pick"] for candidate in candidates]
    visualizer = grasps.GraspFilterVisualizer()
    visualizer.visualize_grasps(grasp_poses=poses, results=filter_results, planning_scene=scene)
    visualizer.visualize_pregrasp_poses(
        grasp_poses=poses,
        results=filter_results,
        planning_scene=scene,
        pre_grasp_offset_m=PRE_GRASP_OFFSET_M,
    )
    recording.log(
        "/grasp_planner/candidates",
        rr.Points3D(
            positions=[pose[:3, 3] for pose in poses],
            colors=[(0, 200, 90) if plan.accepted else (220, 60, 60) for plan in plans],
            labels=[
                f"candidate {index}: {'accepted' if plan.accepted else 'rejected'}"
                for index, plan in enumerate(plans)
            ],
        ),
        static=True,
    )

    # Each segment holds joint positions in degrees. A rejected plan may still hold a
    # partial path, which is useful to inspect. Candidates the filter rejected have none.
    sequence = 0
    for plan in plans:
        for segment_name, trajectory in (
            ("APPROACH", plan.approach_joint_trajectory_deg),
            ("LIFT", plan.lift_joint_trajectory_deg),
            ("RETREAT", plan.retreat_joint_trajectory_deg),
        ):
            configuration = open_configuration if segment_name == "APPROACH" else closed_configuration
            scene.set_attached_object_configuration("gripper", configuration)
            for step, joints_deg in enumerate(trajectory):
                recording.set_time("grasp_planning_step", sequence=sequence)
                sequence += 1
                scene.visualize_configuration_rerun(joints_deg, [])
                recording.log(
                    "/grasp_planner/status",
                    rr.TextLog(
                        f"candidate={plan.candidate_index} "
                        f"{'ACCEPTED' if plan.accepted else 'REJECTED'} "
                        f"segment={segment_name} step={step + 1}/{len(trajectory)} "
                        f"joints_deg={np.round(joints_deg, 2).tolist()}"
                    ),
                )
    scene.set_attached_object_configuration("gripper", open_configuration)
    recording.flush()
    if save_rrd:
        logger.info(f"Saved Rerun recording: {save_rrd}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Synapse grasp planner example")
    parser.add_argument("--save-rrd", type=str, default=None,
                        help="Save the Rerun recording to this .rrd file instead of opening the viewer")
    args = parser.parse_args()

    main(save_rrd=args.save_rrd)
