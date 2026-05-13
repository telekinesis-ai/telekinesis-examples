"""
Set Cartesian Pose with rerun visualization example for the Synapse SDK.

This example runs against real robot hardware. ``set_cartesian_pose`` is
defined on the abstract manipulator and supports all robots that implement
the hardware backend; a Universal Robots (UR10e) is used here purely for
illustration. The live robot state and target TCP frames are streamed to a
rerun viewer.

Install:
    pip install rerun-sdk==0.31

Usage:
    python set_cartesian_pose_with_visualization_example.py --ip <ROBOT_IP> --list
    python set_cartesian_pose_with_visualization_example.py --ip <ROBOT_IP> --example <NAME>
    python set_cartesian_pose_with_visualization_example.py --ip <ROBOT_IP> --all
"""

import argparse
import difflib
import time

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


def _visualize_target_frame(path: str, pose, axis_length: float = 0.1) -> None:
    """Draw a 6-vec pose [x, y, z, rx, ry, rz] (deg) as RGB axis arrows."""
    transformation_mat = utils.pose_to_transformation_matrix(pose, "deg")
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


def set_cartesian_pose_with_visualization_example(ip: str):
    """Move TCP -2 cm then +2 cm in Z with a live rerun visualization. Supports all robots."""

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

        # ----- Example 1: synchronous -2 cm in Z -----
        initial = robot.get_cartesian_pose()
        target_1 = list(initial)
        target_1[2] -= 0.02
        _visualize_target_frame("/target_pose_1", target_1)

        # Command the Cartesian move and refresh the visualization
        robot.set_cartesian_pose(cartesian_pose=target_1,
                                 speed=0.25,
                                 acceleration=0.25,
                                 asynchronous=False)
        _visualize_robot(robot)
        logger.success(f"Moved to {target_1}")

        # ----- Example 2: synchronous +2 cm in Z -----
        current = robot.get_cartesian_pose()
        target_2 = list(current)
        target_2[2] += 0.02
        _visualize_target_frame("/target_pose_2", target_2)

        # Command the Cartesian move and refresh the visualization
        robot.set_cartesian_pose(cartesian_pose=target_2,
                                 speed=0.1,
                                 acceleration=0.1,
                                 asynchronous=False)
        _visualize_robot(robot)
        logger.success(f"Moved to {target_2}")

    # Ensure we stop the robot and disconnect even if there was an error
    finally:
        robot.disconnect()


def get_example_dict(ip: str):
    return {
        "set_cartesian_pose_with_visualization": lambda: set_cartesian_pose_with_visualization_example(ip),
    }


def main():
    """
    Run a Set Cartesian Pose with rerun visualization Synapse example.
    Usage:
        python set_cartesian_pose_with_visualization_example.py --ip <ROBOT_IP> --list
        python set_cartesian_pose_with_visualization_example.py --ip <ROBOT_IP> --example <NAME>
        python set_cartesian_pose_with_visualization_example.py --ip <ROBOT_IP> --all
    """

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Set Cartesian Pose with rerun visualization Synapse example")
    parser.add_argument("--ip", type=str, required=True, help="Robot IP address")
    parser.add_argument("--example", type=str)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()
    examples = get_example_dict(ip=args.ip)

    # Handle example selection
    if args.list:
        for name in sorted(examples):
            logger.info(f"  - {name}")
        return
    if args.all:
        for name, fn in examples.items():
            logger.info(f"Running {name}...")
            try:
                fn()
            except Exception as e:
                logger.error(f"{name} FAILED: {type(e).__name__}: {e}")
        return

    # Handle single example execution
    if not args.example:
        logger.error("Provide --example, --list, or --all.")
        raise SystemExit(1)
    name = args.example.lower()
    if name not in examples:
        matches = difflib.get_close_matches(name, examples.keys(), n=3, cutoff=0.4)
        logger.error(f"Example '{name}' not found.")
        if matches:
            logger.error("Did you mean: " + ", ".join(matches))
        raise SystemExit(1)
    examples[name]()


if __name__ == "__main__":
    main()
