"""
Example: move to 3 world-frame poses using cartesian control, with both a
default and a custom TCP frame -- offline, with live Rerun logging.

No hardware required -- runs entirely on the kinematic model. A babyros
subscriber redraws the robot in Rerun continuously as it moves.

Demonstrates:
  - ``robot.attach_tool()``        -- attach a gripper
  - ``robot.add_tcp()``            -- register a custom TCP offset
  - ``robot.set_cartesian_pose()`` -- move via world-frame target
  - ``robot.active_tcp``           -- switch the active end-effector frame

Run:
    python set_cartesian_pose_with_custom_tcp_rerun_live.py
"""

import time
from functools import partial

import numpy as np
import rerun as rr
from loguru import logger
from babyros import node

from telekinesis.tf import tftree, tfutils
from telekinesis.synapse.robots.manipulators import universal_robots
from telekinesis.synapse.tools.parallel_grippers import onrobot


def interpolated_poses(start: list, end: list, n_steps: int) -> list:
    """Return a list of ``n_steps`` linearly interpolated poses from ``start`` to ``end``."""
    start_arr = np.array(start, dtype=float)
    end_arr = np.array(end, dtype=float)
    return [
        (start_arr + (step / n_steps) * (end_arr - start_arr)).tolist()
        for step in range(1, n_steps + 1)
    ]


def on_state(msg, robot, recording):
    """Redraw the robot in Rerun on each state message.

    ``robot`` and ``recording`` are bound via functools.partial so the callback
    keeps babyros's single-argument (msg) signature. Passing ``recording``
    explicitly is required because this callback runs on a babyros worker thread,
    where Rerun's thread-local active recording is not set -- without it,
    ``visualize_rerun`` would spawn a new viewer per message.
    """
    robot.visualize_rerun(recording_stream=recording)


def main():

    # Create robot with a name so its state publisher starts
    # (the subscriber below needs a non-empty state_publisher_topic).
    robot = universal_robots.UniversalRobotsUR10E(name="manipulator1")

    # Attach a gripper and add a custom TCP at its tip
    gripper = onrobot.OnRobotRG6()
    robot.attach_tool(gripper)
    robot.add_tcp(name="gripper_tip",
                    transform=[0.0, 0.0, 0.25, 0.0, 0.0, 0.0],
                    set_active=False)
    

    # Visualize the robot in Rerun, and capture the recording stream so the
    # subscriber callback can draw into it from its worker thread.
    robot.visualize_rerun()
    recording = rr.get_global_data_recording()

    # Home pose and target cartesian poses
    home_cartesian_pose = [0.40, 0.00, 0.60, 180.0, 0.0, 0.0]
    target_cartesian_poses = [
        [0.40,  0.10,  0.45, 180.0, 0.0, 0.0],
        [0.40, -0.10,  0.45, 180.0, 0.0, 0.0],
        [0.35,  0.00,  0.55, 180.0, 0.0, 0.0],
    ]

    # Build and draw a TF tree of the home and target frames in Rerun
    tree = tftree.TransformTree("world")
    tree.add("world", "home",
             tfutils.pose_to_transformation_matrix(home_cartesian_pose, rot_type="deg"),
             rot_type="mat")
    for i, target in enumerate(target_cartesian_poses):
        tree.add("world", f"target_{i + 1}",
                 tfutils.pose_to_transformation_matrix(target, rot_type="deg"),
                 rot_type="mat")
    tree.visualize_rerun(axis_len=0.05, recording_stream=recording)

    # Subscriber redraws the robot in Rerun as it moves
    sub = node.Subscriber(topic=robot.state_publisher_topic,
                          callback=partial(on_state, robot=robot, recording=recording))

    try:
        # Section 1: cartesian control with the default (tool0) TCP active.
        # Establish a known start at home, tour the targets, then return home.
        logger.info(f"--- Section 1: cartesian control (active TCP: '{robot.active_tcp}') ---")

        hz = 60
        n_steps = 40
        for target in [home_cartesian_pose, *target_cartesian_poses, home_cartesian_pose]:
            for pose in interpolated_poses(robot.get_cartesian_pose(), target, n_steps):
                robot.set_cartesian_pose(pose)
                time.sleep(1.0 / hz)

        # Section 2: tour the targets and return home with the gripper tip active
        robot.active_tcp = "gripper_tip"
        logger.info(f"--- Section 2: cartesian control (active TCP: '{robot.active_tcp}') ---")
        for target in [*target_cartesian_poses, home_cartesian_pose]:
            for pose in interpolated_poses(robot.get_cartesian_pose(), target, n_steps):
                robot.set_cartesian_pose(pose)
                time.sleep(1.0 / hz)

    finally:
        sub.delete()


if __name__ == "__main__":
    main()
