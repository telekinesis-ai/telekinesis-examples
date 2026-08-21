"""
Telekinesis quickstart: drive a Fanuc robot along a YZ-plane circle via Cartesian pose targets.
No Hardware Required - runs entirely in software with live visualization in Rerun.

Traces a closed circle of radius 0.20m in the YZ plane around the home TCP pose. The TCP
path is drawn live as a connected line with a hue gradient (older
segments blue, newest red).

Run:
    python examples/synapse/quickstart_set_cartesian_pose_fanuc.py
"""

import colorsys
import time

import numpy as np
import rerun as rr
from loguru import logger

from telekinesis.synapse.robots.manipulators import fanuc


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
    """Trace a YZ-plane circle around the Fanuc's home TCP pose, visualized in rerun."""

    # Frequency to update the visualization (Hz)
    hz = 30
    dt = 1.0 / hz

    # Radius of the circle to trace (meters)
    radius = 0.2
    n_steps = 200

    # Create robot
    robot = fanuc.FanucM10IA()

    # Initialize rerun and log static meshes
    rr.init(f"telekinesis_synapse_{type(robot).__name__}", spawn=True)
    robot.visualize_rerun(axis_length=0.1, recording_stream=rr.get_global_data_recording())
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
        robot.visualize_rerun()
        actual = robot.get_cartesian_pose()
        path.append([float(actual[0]), float(actual[1]), float(actual[2])])
        visualize_path(path)

        # Sleep to maintain a consistent visualization rate.
        time.sleep(dt)


if __name__ == "__main__":
    main()
