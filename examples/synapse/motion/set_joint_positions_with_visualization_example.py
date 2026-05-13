"""
Set Joint Positions with rerun visualization example for the Synapse SDK.

This example runs against real robot hardware. The live robot state and
target TCP frames (derived from forward kinematics of the target joint
configuration) are streamed to a rerun viewer.

Currently supported only for Universal Robots (UR10e).

For offline, refer to quick start examples.

Install:
    pip install rerun-sdk==0.31

Usage:
    python set_joint_positions_with_visualization_example.py --ip <ROBOT_IP>
"""

import argparse
import time

import numpy as np
import rerun as rr
from loguru import logger

from telekinesis.synapse import utils
from telekinesis.synapse.robots.manipulators import universal_robots


def _visualize_robot(robot, static_meshes: bool = False) -> None:
    """Log per-link transforms to rerun, plus the static meshes on the first call."""
    if static_meshes:
        for link, m in robot.get_visual_meshes_data().items():
            if m["vertices"] is None:
                continue
            kwargs: dict = {
                "vertex_positions": m["vertices"],
                "triangle_indices": m["triangles"],
                "vertex_normals": m["vertex_normals"],
            }
            if m["vertex_colors"] is not None:
                kwargs["vertex_colors"] = m["vertex_colors"]
            else:
                kwargs["albedo_factor"] = m["color"] or [179, 179, 179]
            rr.log(f"/robot/{link}", rr.Mesh3D(**kwargs), static=True)

    for link, transformation_mat in robot.get_visual_mesh_transforms().items():
        rr.log(
            f"/robot/{link}",
            rr.Transform3D(
                translation=transformation_mat[:3, 3],
                mat3x3=transformation_mat[:3, :3],
            ),
        )


def _visualize_target_joints(robot, path: str, joint_positions, axis_length: float = 0.1) -> None:
    """Forward-kinematic the target joint config and draw the resulting TCP frame."""
    target_pose = robot.forward_kinematics(joint_positions)
    transformation_mat = np.asarray(
        utils.pose_to_transformation_matrix(target_pose, rot_type="deg")
    )
    origin = transformation_mat[:3, 3]
    rr.log(
        path,
        rr.Arrows3D(
            origins=[origin, origin, origin],
            vectors=[
                transformation_mat[:3, 0] * axis_length,
                transformation_mat[:3, 1] * axis_length,
                transformation_mat[:3, 2] * axis_length,
            ],
            colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
        ),
    )


def main(ip: str):
    """Move joint 0 by ±60 deg with a live rerun visualization. Supports all robots."""

    # Create the robot, initialize rerun, and log the static meshes
    robot = universal_robots.UniversalRobotsUR10E()
    rr.init(f"telekinesis_synapse_{type(robot).__name__}", spawn=True)
    _visualize_robot(robot, static_meshes=True)
    time.sleep(2.0)

    # Connect to the robot and refresh the visualization with the live state
    robot.connect(ip=ip)
    _visualize_robot(robot)

    # Run two synchronous moves and re-log the robot state after each
    try:

        # ----- Example 1: synchronous +60 deg on joint 0 -----
        delta_deg = 60.0
        initial = robot.get_joint_positions()
        q_target_1 = list(initial)
        q_target_1[0] += delta_deg
        _visualize_target_joints(robot, "/target_joints_1", q_target_1)

        # Command the joint move and refresh the visualization
        robot.set_joint_positions(joint_positions=q_target_1,
                                  speed=20,
                                  acceleration=20,
                                  asynchronous=False)
        _visualize_robot(robot)
        logger.success(f"Moved to {q_target_1}")

        # ----- Example 2: synchronous -60 deg on joint 0 -----
        current = robot.get_joint_positions()
        q_target_2 = list(current)
        q_target_2[0] -= delta_deg
        _visualize_target_joints(robot, "/target_joints_2", q_target_2)

        # Command the joint move and refresh the visualization
        robot.set_joint_positions(joint_positions=q_target_2,
                                  speed=20,
                                  acceleration=20,
                                  asynchronous=False)
        _visualize_robot(robot)
        logger.success(f"Moved to {q_target_2}")

    # Ensure we stop the robot and disconnect even if there was an error
    finally:
        robot.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Set Joint Positions with rerun visualization Synapse example"
    )
    parser.add_argument("--ip", type=str, required=True, help="Robot IP address")
    args = parser.parse_args()

    main(ip=args.ip)
