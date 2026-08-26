"""
Telekinesis quickstart: drive a Neura Robotics robot along a YZ-plane circle via Cartesian pose targets.
No Hardware Required - runs entirely in software with live visualization in Rerun.
--prim_path is accepted for testing, but Isaac Sim is not yet implemented for Neura Robotics
in this SDK version, so connect(simulation_prim_path=...) will raise.

Traces a closed circle of radius 0.30m in the YZ plane around the home TCP pose. The TCP
path is drawn live as a connected line with a hue gradient (older
segments blue, newest red).

Run:
    python quickstart_set_cartesian_pose_neura_robotics.py
    python quickstart_set_cartesian_pose_neura_robotics.py --prim_path <PRIM_PATH>
"""

import argparse
import colorsys

import numpy as np
import rerun as rr

from telekinesis.synapse.robots.manipulators import neura_robotics


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


def main(prim_path: str | None) -> None:
    """Trace a YZ-plane circle around the MAiRA7M's home TCP pose, visualized in rerun."""

    # =========================== Create Robot ==================================
    robot = neura_robotics.NeuraRoboticsMAiRA7M(name="NeuraRoboticsMAiRA7M")

    # =========================== Visualization (Optional) =============================
    robot.visualize_rerun()

    try:
        #===================== Connect Robot (Optional) ============================
        if prim_path:
            robot.connect(simulation_prim_path=prim_path)
            robot.set_joint_positions(robot.default_joint_configuration)  # Move to default pose in simulation

        # ========================== Draw Circle ====================================
        radius = 0.30
        n_steps = 50

        home_pose = robot.get_cartesian_pose()
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

            actual = robot.get_cartesian_pose()
            path.append([float(actual[0]), float(actual[1]), float(actual[2])])
            visualize_path(path)
    finally:
        if prim_path:
            robot.disconnect()
        robot.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Drive a Neura Robotics robot along a circle"
    )
    parser.add_argument("--prim_path", type=str, default=None,
                         help='Isaac Sim articulation prim path, e.g. "/World/maira7m"')
    args = parser.parse_args()

    main(prim_path=args.prim_path)
