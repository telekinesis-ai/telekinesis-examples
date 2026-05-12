"""Drive a real UR10e through advanced Cartesian moves, with a live Rerun feed.

Connects to a UR10e, then runs two example moves:
  - Example 1: synchronous move -20cm in Z (blocks until complete).
  - Example 2: asynchronous move +20cm in Z, stopped after 0.5s with
    ``stop_cartesian_motion`` before it reaches the target.

Each example draws the target TCP frame in rerun (RGB axis arrows) and
the live robot state is streamed throughout the async move.

Make sure the cell is clear and the robot is in remote-control mode
before running.

Install:
    pip install rerun-sdk==0.31  # tested on 0.31

Run (replace ``192.168.x.y`` with your UR10e or URSim IP):
    python examples/py/ur10e_set_cartesian_pose_advanced_with_visualization.py --ip 192.168.x.y
"""

import argparse
import time

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


def visualize_target_frame(path: str, pose, axis_length: float = 0.1) -> None:
    """Draw a 6-vec pose [x, y, z, rx, ry, rz] (deg) as RGB axis arrows.

    Columns of the rotation matrix are the world-frame X/Y/Z axis vectors.
    Red -> X, Green -> Y, Blue -> Z.
    """
    transformation_mat = utils.pose_to_transformation_matrix(pose, "deg")
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
    """Connect to a UR10e and run the synchronous + asynchronous Cartesian demos."""

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
        # ----- Example 1: synchronous move -20cm in Z -----
        delta_z = 0.2
        initial_tcp_pose = robot.get_cartesian_pose()
        new_tcp_pose = initial_tcp_pose[:]
        new_tcp_pose[2] -= delta_z
        visualize_target_frame("/target_pose_1", new_tcp_pose)

        # Move the robot
        robot.set_cartesian_pose(
            cartesian_pose=new_tcp_pose,
            speed=0.25,
            acceleration=0.25,
            asynchronous=False,
        )
        visualize_robot(robot)
        logger.info(f"Moved to target Cartesian pose: {new_tcp_pose}")

        # ----- Example 2: asynchronous move +20cm in Z, stopped after 0.5s -----
        actual_tcp_pose = robot.get_cartesian_pose()
        new_tcp_pose = actual_tcp_pose[:]
        new_tcp_pose[2] += delta_z
        visualize_target_frame("/target_pose_2", new_tcp_pose)

        # Move the robot
        robot.set_cartesian_pose(
            cartesian_pose=new_tcp_pose,
            speed=0.1,
            acceleration=0.1,
            asynchronous=True,
        )
        time.sleep(2)
        robot.stop_cartesian_motion(stopping_speed=0.25)

        logger.info(f"Stopped Cartesian motion before reaching target Cartesian pose: {new_tcp_pose}")
        visualize_robot(robot)

    finally:
        # Disconnect the robot
        logger.info("Disconnecting from UR10e...")
        robot.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UR10e advanced Cartesian example with Rerun viz")
    parser.add_argument(
        "--ip",
        type=str,
        required=True,
        help="IPv4 address (or hostname) of the UR10e controller",
    )
    args = parser.parse_args()

    main(args.ip)
