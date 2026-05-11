"""Visualize a Franka Panda in Rerun via joint-position targets. No hardware required.

Sweeps the base 360° around home while a secondary joint oscillates ±30°. The TCP
traces the resulting wavy path, drawn live as a connected line with a hue gradient
(older segments blue, newest red).

Install:
    pip install rerun-sdk==0.31  # tested on 0.31

Run:
    python examples/py/github_examples/franka_robotics_set_joint_positions_with_visualization.py
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
    """Sweep the Panda's base while a secondary joint oscillates, visualized in rerun."""

    # Frequency to update the visualization (Hz)
    hz = 30
    dt = 1.0 / hz

    # Base motion parameters (deg, deg/s)
    base_joint_span = 360.0
    base_joint_speed = 60.0

    # Secondary-joint oscillation parameters (deg)
    elbow_amplitude_deg = 30.0
    elbow_cycles = 4

    # Total number of waypoints in trajectory
    n_steps = int(base_joint_span / (base_joint_speed * dt))

    # Create robot
    robot = franka_robotics.FrankaRoboticsPanda()

    # Initialize rerun and log static meshes
    rr.init(f"telekinesis_synapse_{type(robot).__name__}", spawn=True)
    visualize_robot(robot, static_meshes=True)
    time.sleep(2.0)

    # Home configuration to sweep around
    home_q = np.asarray(robot.get_joint_positions(), dtype=float)
    logger.info(
        f"Base {base_joint_span:.0f}° + secondary ±{elbow_amplitude_deg:.0f}° ({n_steps} steps)"
    )

    # Robot motion: sweep base, oscillate secondary joint, visualize robot and TCP path
    path: list[list[float]] = []

    for step in range(n_steps + 1):
        # Normalised progress through the sweep, 0 -> 1.
        t = step / n_steps

        # Centre the base sweep on home so it stays inside symmetric joint limits.
        q = home_q.copy()
        q[0] += base_joint_span * (t - 0.5)
        # 7-DOF arm: elbow lives at q[3] (q[2] is the upper-arm twist).
        q[3] += elbow_amplitude_deg * np.sin(2.0 * np.pi * elbow_cycles * t)

        # Move the robot
        try:
            robot.set_joint_positions(q.tolist())
        except ValueError:
            continue  # outside joint limits

        # Visualize robot and path
        visualize_robot(robot)
        pose = robot.get_cartesian_pose()
        path.append([float(pose[0]), float(pose[1]), float(pose[2])])
        visualize_path(path)

        # Sleep to maintain a consistent visualization rate.
        time.sleep(dt)


if __name__ == "__main__":
    main()
