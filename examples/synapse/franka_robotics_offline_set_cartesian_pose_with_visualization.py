"""Visualize a Franka Panda in Rerun via Cartesian-pose targets. No hardware required.

Traces a closed circle of radius 0.10m in the YZ plane around the home TCP pose. The TCP
path is drawn live as a connected line with a hue gradient (older
segments blue, newest red).

Install:
    pip install rerun-sdk==0.31  # tested on 0.31

Run:
    python examples/py/github_examples/franka_robotics_set_cartesian_pose_with_visualization.py
"""

import colorsys
import time

import numpy as np
import rerun as rr
from loguru import logger

from telekinesis.synapse.robots.manipulators import franka_robotics


def visualize_robot(robot, static_meshes: bool = False) -> None:
    """Log per-link transforms to rerun, plus the static meshes on the first call."""

    # Log static meshes once
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

    # Log per-link transforms on every update
    for link, T in robot.get_visual_mesh_transforms().items():
        rr.log(f"/robot/{link}", rr.Transform3D(translation=T[:3, 3], mat3x3=T[:3, :3]))


def visualize_path(path: list[list[float]], entity: str = "/trajectory") -> None:
    """Draw the TCP path as connected segments with a blue→red hue gradient."""

    if len(path) < 2:
        return
    segments = [[path[i], path[i + 1]] for i in range(len(path) - 1)]
    n = max(1, len(segments) - 1)
    colors = [
        [int(c * 255) for c in colorsys.hsv_to_rgb((1.0 - i / n) * (240.0 / 360.0), 1.0, 1.0)]
        for i in range(len(segments))
    ]
    rr.log(entity, rr.LineStrips3D(segments, colors=colors, radii=0.003))


def main():
    """Trace a YZ-plane circle around the Panda's home TCP pose, visualized in rerun."""

    # Frequency to update the visualization (Hz)
    hz = 20
    dt = 1.0 / hz

    # Radius of the circle to trace (meters)
    radius = 0.10
    n_steps = 200

    # Create robot
    robot = franka_robotics.FrankaRoboticsPanda()

    # Initialize rerun and log static meshes
    rr.init(f"telekinesis_synapse_{type(robot).__name__}", spawn=True)
    visualize_robot(robot, static_meshes=True)
    time.sleep(2.0)

    # Get home pose (default configuration)
    home_pose = robot.get_cartesian_pose()
    logger.info(f"Tracing circle of radius {radius:.3f} m in YZ plane ({n_steps} steps)")

    # Robot motion: draw circle in YZ plane, visualize robot and TCP path
    path: list[list[float]] = []

    for step in range(n_steps + 1):
        theta = 2.0 * np.pi * step / n_steps

        # Circle in the YZ plane, offset so it "kisses" the home pose at theta=0.
        pose = home_pose.copy()
        pose[1] = home_pose[1] + radius * np.cos(theta) - radius
        pose[2] = home_pose[2] + radius * np.sin(theta)

        # Move the robot
        try:
            robot.set_cartesian_pose(pose)
        except ValueError:
            continue  # outside reach / joint limits

        # Visualize robot and path
        visualize_robot(robot)
        actual = robot.get_cartesian_pose()
        path.append([float(actual[0]), float(actual[1]), float(actual[2])])
        visualize_path(path)

        # Sleep to maintain a consistent visualization rate.
        time.sleep(dt)


if __name__ == "__main__":
    main()
