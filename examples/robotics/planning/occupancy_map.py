"""
Builds an occupancy map from a synthetic point cloud, removes the points that land on the
robot, installs the map in a planning scene, and checks the robot for collisions,
visualized in Rerun.

Supports Universal Robots (UR), Epson, and virtual. Runs offline, no hardware needed.

Usage:
    python occupancy_map.py
    python occupancy_map.py --save-rrd occupancy_map.rrd
"""

import argparse

import numpy as np
import rerun as rr
from loguru import logger

from telekinesis.synapse import occupancy_map, planning_scene
from telekinesis.synapse.robots.manipulators import universal_robots

JOINTS_DEG = [0.0, -90.0, 90.0, 0.0, 90.0, 0.0]
MAP_RESOLUTION_M = 0.02


def init_rerun(app_id: str, save_rrd: str | None) -> "rr.RecordingStream":
    """Start a Rerun recording, live in the viewer or saved to an .rrd file."""
    rr.init(app_id, spawn=save_rrd is None)
    recording = rr.get_global_data_recording()
    if save_rrd:
        recording.save(save_rrd)
    return recording


def main(save_rrd: str | None) -> None:
    """Scan a synthetic wall plus points on the robot, then collide against the map."""

    #===================== Create Robot and Scene ================================
    robot = universal_robots.UniversalRobotsUR10E(name='UR10e')
    scene = planning_scene.PlanningScene(kinematic_model=robot.kinematic_model)

    # ==================== Visualization ========================================
    recording = init_rerun("telekinesis_synapse_occupancy_map", save_rrd)
    recording.log("/", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    # Installing a map below refreshes the scene view, and every collision check is drawn.
    scene.visualize_rerun(recording_stream=recording)

    #===================== Create the Map and Its Updater ========================
    occupancy = occupancy_map.OccupancyMapMonitor(MAP_RESOLUTION_M)
    occupancy.add_observer(lambda version: logger.info(f"Map updated, version {version}"))

    # raycast=False only marks the voxel of each point. Use it when the map is rebuilt
    # from scratch for every scan; keep the default True for a map that accumulates scans.
    updater = occupancy_map.PointCloudOctomapUpdater(raycast=False)
    occupancy.add_updater(updater)

    # Connect the scene's robot geometry, so scan points on the robot are removed.
    planning_scene.PlanningSceneMonitor(scene, occupancy, shape_padding=0.05)

    # ==================== Run Skill ============================================
    # A synthetic wall of points 0.9 m in front of the base, in the world frame.
    ys, zs = np.meshgrid(np.linspace(-0.2, 0.2, 21), np.linspace(0.3, 0.6, 16))
    wall = np.column_stack([np.full(ys.size, 0.9), ys.ravel(), zs.ravel()])

    # Points at the TCP, which sit on the robot and should be masked out.
    world_T_tcp = robot.kinematic_model.forward_kinematics(
        np.deg2rad(JOINTS_DEG), robot.default_tcp
    )
    on_robot = np.tile(world_T_tcp[:3, 3], (10, 1))
    points = np.vstack([wall, on_robot])

    inserted = updater.update_world_points(
        points,
        world_T_sensor=np.eye(4),
        rot_type="mat",
        joint_positions_deg=JOINTS_DEG,
    )
    logger.info(f"Scan points: {len(points)}, inserted after robot masking: {inserted}")
    logger.info(f"Accepted points shape: {updater.last_filtered_world_points.shape}")

    # The raw scan in blue, the points that sit on the robot in red, and the points the updater
    # accepted into the map in green.
    recording.log("/scan/wall", rr.Points3D(wall, colors=[[31, 119, 180]], radii=0.006), static=True)
    recording.log("/scan/on_robot", rr.Points3D(on_robot, colors=[[214, 39, 40]], radii=0.012), static=True)
    recording.log(
        "/scan/accepted",
        rr.Points3D(updater.last_filtered_world_points, colors=[[44, 160, 44]], radii=0.004),
        static=True,
    )

    # Dilate occupied voxels with a 1 cm spherical kernel (rounded up to whole voxels),
    # then freeze an independent snapshot.
    occupancy.apply_morphology(0.01)
    snapshot = occupancy.freeze()

    scene.set_occupancy_map(snapshot, collision_margin=0.005)
    logger.info(f"Scene has map: {scene.has_octomap}, nodes: {scene.octomap_size}, "
                f"resolution: {scene.octomap_resolution} m")
    recording.set_time("check", sequence=0)
    logger.info(f"Colliding with map: {scene.is_robot_colliding(JOINTS_DEG)}")

    #===================== ShapeMask on Its Own ==================================
    # Test which points lie inside the padded robot shapes at this configuration.
    mask = occupancy_map.ShapeMask(padding=0.05)
    shape_poses = {}
    for geometry in scene.get_geometries_from_planning_scene(JOINTS_DEG):
        shape_poses[mask.add_shape(geometry)] = geometry.world_T_geometry
    inside = mask.mask_containment(points, shape_poses)
    logger.info(f"Points inside robot shapes: {int(inside.sum())} of {len(points)}")

    recording.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Occupancy map Synapse example")
    parser.add_argument("--save-rrd", type=str, default=None,
                        help="Save the Rerun recording to this .rrd file instead of opening the viewer")
    args = parser.parse_args()

    main(save_rrd=args.save_rrd)
