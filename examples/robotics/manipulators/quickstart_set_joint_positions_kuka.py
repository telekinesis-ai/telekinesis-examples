"""
Telekinesis quickstart: drive a KUKA robot in joint space along a sweeping trajectory.
No Hardware Required - runs entirely in software with live visualization in Rerun.
--prim_path is accepted for testing, but Isaac Sim is not yet implemented for KUKA
in this SDK version, so connect(simulation_prim_path=...) will raise.

Sweeps the base 360° around home while a secondary joint oscillates ±30°. The TCP
traces the resulting wavy path, drawn live as a connected line with a hue gradient
(older segments blue, newest red).

Run:
    python quickstart_set_joint_positions_kuka.py
    python quickstart_set_joint_positions_kuka.py --prim_path <PRIM_PATH>
"""

import argparse
import colorsys

import numpy as np
import rerun as rr

from telekinesis.synapse.robots.manipulators import kuka


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
    """Sweep the KUKA's base while a secondary joint oscillates, visualized in rerun."""

    # =========================== Create Robot ==================================
    robot = kuka.KukaKR1502(name="KukaKR1502")

    # =========================== Visualization (Optional) =============================
    robot.visualize_rerun()

    try:
        #===================== Connect Robot (Optional) ============================
        if prim_path:
            robot.connect(simulation_prim_path=prim_path)
            robot.set_joint_positions(robot.default_joint_configuration)  # Move to default pose in simulation

        # ========================== Sweep Trajectory ====================================
        base_joint_span = 360.0
        elbow_amplitude_deg = 30.0
        elbow_cycles = 4
        n_steps = 180

        home_q = np.asarray(robot.get_joint_positions(), dtype=float)
        path: list[list[float]] = []
        for step in range(n_steps + 1):
            # Normalised progress through the sweep, 0 -> 1.
            t = step / n_steps

            # Centre the base sweep on home so it stays inside symmetric joint limits.
            q = home_q.copy()
            q[0] += base_joint_span * (t - 0.5)
            q[2] += elbow_amplitude_deg * np.sin(2.0 * np.pi * elbow_cycles * t)

            # Move the robot
            try:
                robot.set_joint_positions(q.tolist())
            except ValueError:
                continue  # outside joint limits

            pose = robot.get_cartesian_pose()
            path.append([float(pose[0]), float(pose[1]), float(pose[2])])
            visualize_path(path)
    finally:
        if prim_path:
            robot.disconnect()
        robot.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Drive a KUKA robot in joint space along a sweeping trajectory"
    )
    parser.add_argument("--prim_path", type=str, default=None,
                         help='Isaac Sim articulation prim path, e.g. "/World/kuka_kr1502"')
    args = parser.parse_args()

    main(prim_path=args.prim_path)
