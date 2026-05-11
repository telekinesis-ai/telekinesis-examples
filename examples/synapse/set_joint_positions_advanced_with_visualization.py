"""Drive a real UR10e through advanced joint-position moves, with a live Rerun feed.

Connects to a UR10e (or URSim) over RTDE, then runs two example moves:
  - Example 1: synchronous move of -5° on joint 0 (blocks until complete).
  - Example 2: synchronous move of +5° on joint 0 (blocks until complete).

Each example draws the target TCP frame in rerun (RGB axis arrows) by
computing forward kinematics at the target joint configuration before
the move. The robot mesh is re-logged after each move so the live state
mirrors what the controller is actually reporting.

Make sure the cell is clear and the robot is in remote-control mode
before running.

Install:
    pip install rerun-sdk==0.31  # tested on 0.31

Run (replace ``192.168.x.y`` with your UR10e or URSim IP):
    python examples/py/ur10e_set_joint_positions_advanced_with_visualization.py --ip 192.168.x.y
"""

import argparse
import time

import numpy as np
import rerun as rr
from loguru import logger

from telekinesis.synapse import utils
from telekinesis.synapse.robots.manipulators import universal_robots


def visualize_robot(robot, static_meshes: bool = False) -> None:
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
                translation=transformation_mat[:3, 3], mat3x3=transformation_mat[:3, :3]
            ),
        )


def visualize_target_joints(robot, path: str, joint_positions, axis_length: float = 0.1) -> None:
    """Forward-kinematic the target joint config and draw the resulting TCP frame.

    Columns of the rotation are the world-frame X/Y/Z axes.
    Red -> X, Green -> Y, Blue -> Z.
    """
    # forward_kinematics now consumes degrees and returns a 6-vec deg-Euler
    # pose; convert to a 4x4 matrix for axis extraction.
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
                transformation_mat[:3, 0] * axis_length,  # X axis
                transformation_mat[:3, 1] * axis_length,  # Y axis
                transformation_mat[:3, 2] * axis_length,  # Z axis
            ],
            colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
        ),
    )


def main(robot_ip: str):
    """Connect to a UR10e and run two synchronous joint-position demos."""

    # Create the robot instance
    robot = universal_robots.UniversalRobotsUR10E()

    # Initialize rerun and log static meshes
    rr.init(f"telekinesis_synapse_{type(robot).__name__}", spawn=True)
    visualize_robot(robot, static_meshes=True)
    time.sleep(2.0)

    # Connect to the robot
    logger.info(f"Connecting to UR10e at {robot_ip}...")
    robot.connect(ip=robot_ip)
    visualize_robot(robot)

    try:
        # ----- Example 1: synchronous move -30° on joint 0 -----
        delta_deg = 5
        initial_joint_positions = robot.get_joint_positions()
        new_joint_positions = initial_joint_positions[:]
        new_joint_positions[0] += delta_deg
        visualize_target_joints(robot, "/target_joints_1", new_joint_positions)

        robot.set_joint_positions(
            joint_positions=new_joint_positions,
            speed=20,
            acceleration=20,
            asynchronous=False,
        )
        visualize_robot(robot)
        logger.info(f"Moved to target joint positions: {new_joint_positions}")

        # ----- Example 2: synchronous move +30° on joint 0 -----
        actual_joint_positions = robot.get_joint_positions()
        new_joint_positions = actual_joint_positions[:]
        new_joint_positions[0] -= delta_deg
        visualize_target_joints(robot, "/target_joints_2", new_joint_positions)

        robot.set_joint_positions(
            joint_positions=new_joint_positions,
            speed=20,
            acceleration=20,
            asynchronous=False,
        )
        visualize_robot(robot)

    finally:
        # Disconnect the robot
        logger.info("Disconnecting from UR10e...")
        robot.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="UR10e advanced joint-positions example with Rerun viz"
    )
    parser.add_argument(
        "--ip",
        type=str,
        required=True,
        help="IPv4 address (or hostname) of the UR10e controller",
    )
    args = parser.parse_args()

    main(args.ip)
