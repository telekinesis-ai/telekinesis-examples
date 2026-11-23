"""
Checks a UR10e against a small occupied voxel, once at the TCP and once far away,
visualized in Rerun.

Supports Universal Robots (UR), Epson, and virtual. Runs offline, no hardware needed.

Usage:
    python collision_checker.py
    python collision_checker.py --save-rrd collision_checker.rrd
"""

import argparse
from types import SimpleNamespace

import numpy as np
import rerun as rr
from loguru import logger

from telekinesis.synapse import collision_checker, occupancy_map, planning_scene
from telekinesis.synapse.robots.manipulators import universal_robots

JOINTS_DEG = [0.0, -90.0, 90.0, 0.0, 90.0, 0.0]
AWAY_JOINTS_DEG = [90.0, -90.0, 90.0, 0.0, 90.0, 0.0]


def init_rerun(app_id: str, save_rrd: str | None) -> "rr.RecordingStream":
    """Start a Rerun recording, live in the viewer or saved to an .rrd file."""
    rr.init(app_id, spawn=save_rrd is None)
    recording = rr.get_global_data_recording()
    if save_rrd:
        recording.save(save_rrd)
    return recording


def main(save_rrd: str | None) -> None:
    """Build an occupancy map and check two joint configurations against it."""

    #===================== Create Robot ==========================================
    robot = universal_robots.UniversalRobotsUR10E(name='UR10e')

    #===================== Build an Occupancy Map ================================
    # One occupied voxel at the TCP, one 3 m away from the robot base.
    world_T_tcp = robot.kinematic_model.forward_kinematics(np.deg2rad(JOINTS_DEG), robot.default_tcp)
    monitor = occupancy_map.OccupancyMapMonitor(0.05)
    monitor.mark_occupied(np.array([world_T_tcp[:3, 3], [3.0, 3.0, 3.0]]))
    octomap = monitor.get_octomap()

    # ==================== Visualization ========================================
    # The checker has no viewer of its own, so a planning scene with the same map and margin
    # draws the robot and the occupied voxels. The checker still answers every query below.
    scene = planning_scene.PlanningScene(
        kinematic_model=robot.kinematic_model,
        octomap=octomap,
        collision_margin=0.005,
    )
    recording = init_rerun("telekinesis_synapse_collision_checker", save_rrd)
    recording.log("/", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    scene.visualize_rerun(recording_stream=recording, retain_collision_checks=True)

    # ==================== Run Skill ============================================
    checker = collision_checker.CollisionChecker(
        robot.kinematic_model,
        octomap,
        collision_margin=0.005,
    )
    for step, (label, joints_deg) in enumerate(
        (("at the TCP voxel", JOINTS_DEG), ("away from the voxels", AWAY_JOINTS_DEG))
    ):
        q_rad = np.deg2rad(joints_deg)
        colliding_names = checker.colliding_geometry_names(q_rad=q_rad)
        logger.info(f"{label}: collision={checker.has_collision(q_rad=q_rad)}, "
                    f"colliding geometry={colliding_names}")

        # Draw this configuration with the checker's colliding shapes highlighted.
        recording.set_time("check", sequence=step)
        scene.visualize_configuration_rerun(joints_deg, colliding_names)

    # The same query with joint positions in degrees, from any object with ``joint_positions``.
    # ``robot.state`` works the same way on a connected robot.
    joint_state = SimpleNamespace(joint_positions=JOINTS_DEG)
    logger.info(f"Collision from joint_state: {checker.has_collision(joint_state=joint_state)}")
    logger.info(f"Checker: nq={checker.nq}, map nodes={checker.octomap_size}, "
                f"resolution={checker.octomap_resolution} m")

    recording.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collision checker Synapse example")
    parser.add_argument("--save-rrd", type=str, default=None,
                        help="Save the Rerun recording to this .rrd file instead of opening the viewer")
    args = parser.parse_args()

    main(save_rrd=args.save_rrd)
