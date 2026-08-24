"""
Telekinesis quickstart: drive a Universal Robots robot in joint space along a sweeping trajectory.
No Hardware Required - runs entirely in software with live visualization in Rerun.

Sweeps the base 360° around home while the elbow oscillates ±30°. The
TCP traces the resulting wavy path, drawn live as a connected line
with a hue gradient (older segments blue, newest red).

Run:
    python examples/robotics/quickstart_set_joint_positions_universal_robots.py
"""

import colorsys
import time

import numpy as np
import rerun as rr
from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots

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

    # Elbow oscillation: amplitude (deg) and number of full cycles per base sweep.
    elbow_amplitude_deg = 30.0
    elbow_cycles = 4

    # Total number of waypoints in trajectory
    n_steps = int(base_joint_span / (base_joint_speed * dt))

    # ----------------------------------------------------------------------------
    #                   Robot setup and rerun initialization
    # ----------------------------------------------------------------------------

    # Initialize telekinesis-synapse UR10e robot
    robot = universal_robots.UniversalRobotsUR10E()

    # Initialize Rerun and log the static meshes once.
    rr.init(f"telekinesis_synapse_{type(robot).__name__}", spawn=True)
    robot.visualize_rerun(axis_length=0.1, recording_stream=rr.get_global_data_recording())
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
        robot.visualize_rerun()

        # Visualize the TCP path
        pose = robot.get_cartesian_pose()
        path.append([float(pose[0]), float(pose[1]), float(pose[2])])
        visualize_path(path)

        # Sleep to maintain a consistent visualization rate.
        time.sleep(dt)


if __name__ == "__main__":
    main()
