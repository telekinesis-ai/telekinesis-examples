"""Visualize a UR10e in Rerun via joint-position targets. No hardware required.

Sweeps the base 320° around home while the elbow oscillates ±30°. The
TCP traces the resulting wavy circle, drawn live as a connected line
with a hue gradient (older segments blue, newest red).

Install:
    pip install rerun-sdk==0.31  # tested on 0.31

Run:
    python examples/py/github_examples/universal_robots_set_joint_positions_with_visualization.py
"""

import colorsys
import time

import numpy as np
import rerun as rr
from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots


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


def main():
    """Spin the UR10e base while the elbow wiggles, and trace the TCP in rerun."""

    # ---------------------------------------------------------------------------
    #                   Initial setup and parameters
    # ---------------------------------------------------------------------------

    # Visualization tick rate
    hz = 30
    dt = 1.0 / hz

    # Base motion parameters (deg, deg/s)
    base_joint_span = 360.0
    base_joint_speed = 60.0

    # Number of full elbow oscillations per base revolution.
    elbow_amplitude_deg = 30.0
    elbow_cycles = 4

    # Total number of waypointsin trajectory
    n_steps = int(base_joint_span / (base_joint_speed * dt))

    # ----------------------------------------------------------------------------
    #                   Robot setup and rerun initialization
    # ----------------------------------------------------------------------------

    # Initialize telekinesis-synapse UR10e robot
    robot = universal_robots.UniversalRobotsUR10E()

    # Initialize Rerun and log the static meshes once.
    rr.init(f"telekinesis_synapse_{type(robot).__name__}", spawn=True)
    visualize_robot(robot, static_meshes=True)
    time.sleep(2.0)

    # ---------------------------------------------------------------------------
    #                   Main loop: update joint positions and log to Rerun
    # ---------------------------------------------------------------------------

    # Home configuration to sweep around
    home_q = np.asarray(robot.get_joint_positions(), dtype=float)
    logger.info(
        f"Base {base_joint_span:.0f}° + elbow ±{elbow_amplitude_deg:.0f}° ({n_steps} steps)"
    )

    # Live TCP path for visualization as a connected line strip with a hue gradient.
    path: list[list[float]] = []

    # Main loop: update joint positions and log to Rerun
    for step in range(n_steps + 1):
        # Normalised progress through the sweep, 0 -> 1.
        t = step / n_steps

        # Centre the base sweep on home so it stays inside symmetric joint limits.
        q = home_q.copy()
        q[0] += base_joint_span * (t - 0.5)
        q[2] += elbow_amplitude_deg * np.sin(2.0 * np.pi * elbow_cycles * t)

        # Move Robot
        try:
            # Set the new joint positions
            robot.set_joint_positions(q.tolist())
        except ValueError:
            # Outside joint limits — skip this waypoint and keep going.
            continue

        # Visualize robot
        visualize_robot(robot)

        # Visualize the TCP path
        pose = robot.get_cartesian_pose()
        path.append([float(pose[0]), float(pose[1]), float(pose[2])])
        if len(path) >= 2:
            segments = [[path[i], path[i + 1]] for i in range(len(path) - 1)]
            colors = []
            for i in range(len(segments)):
                # Hue: blue (240°) for oldest, red (0°) for newest.
                h = (1.0 - i / max(1, len(segments) - 1)) * (240.0 / 360.0)
                r, g, b = colorsys.hsv_to_rgb(h, 1.0, 1.0)
                colors.append([int(r * 255), int(g * 255), int(b * 255)])
            rr.log("/trajectory", rr.LineStrips3D(segments, colors=colors, radii=0.003))

        # Sleep to maintain a consistent visualization rate.
        time.sleep(dt)


if __name__ == "__main__":
    main()
