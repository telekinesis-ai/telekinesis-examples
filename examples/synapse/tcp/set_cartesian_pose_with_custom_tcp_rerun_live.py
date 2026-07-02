"""
Example: move 2 cm along +Z from the current pose, first with the default
(tool0) TCP active and then with a custom TCP active -- with live Rerun logging.

Requires hardware -- connects to the robot at ``--ip`` and executes the motion.
A babyros subscriber redraws the robot in Rerun continuously as it moves.

Demonstrates:
  - ``robot.add_tcp()``            -- register a custom TCP offset
  - ``robot.active_tcp``           -- switch the active end-effector frame
  - ``robot.set_cartesian_pose()`` -- move via world-frame target; verify arrival

Currently supported only for real hardware from Universal Robots.

For an offline version, refer to tcp/offline/set_cartesian_pose_with_custom_tcp_rerun_live.py

Usage:
    python set_cartesian_pose_with_custom_tcp_rerun_live.py [--ip <ROBOT_IP>]
"""

import argparse
from functools import partial

import rerun as rr
from loguru import logger
from babyros import node

from telekinesis.tf import tftree, tfutils
from telekinesis.synapse.robots.manipulators import universal_robots


def on_state(msg, robot, recording):
    """Redraw the robot in Rerun on each state message.

    ``robot`` and ``recording`` are bound via functools.partial so the callback
    keeps babyros's single-argument (msg) signature. Passing ``recording``
    explicitly is required because this callback runs on a babyros worker thread,
    where Rerun's thread-local active recording is not set -- without it,
    ``visualize_rerun`` would spawn a new viewer per message.
    """
    robot.visualize_rerun(recording_stream=recording)


def main(ip: str | None = None):
    # Create the robot with a name so its state publisher starts, then connect.
    robot = universal_robots.UniversalRobotsUR10E(name="manipulator1")
    robot.connect(ip=ip)

    # Visualize the robot in Rerun, and capture the recording stream so the
    # subscriber callback can draw into it from its worker thread.
    robot.visualize_rerun()
    recording = rr.get_global_data_recording()

    # Subscriber redraws the robot in Rerun as it moves
    sub = node.Subscriber(topic=robot.state_publisher_topic,
                          callback=partial(on_state, robot=robot, recording=recording))

    try:
        logger.info(f"Active TCP: {robot.active_tcp}")

        # Target: current pose moved 2 cm along +Z
        target_tcp_pose = list(robot.get_cartesian_pose())
        target_tcp_pose[2] += 0.02

        # Draw the target frame in Rerun as a TF tree
        tree = tftree.TransformTree("world")
        tree.add("world", "target",
                 tfutils.pose_to_transformation_matrix(target_tcp_pose, rot_type="deg"),
                 rot_type="mat")
        tree.visualize_rerun(axis_len=0.05,
                             recording_stream=recording)

        # Move with the default (tool0) TCP active
        robot.set_cartesian_pose(target_tcp_pose)
        logger.success(f"arrived: {robot.get_cartesian_pose()}")

        # Add a custom TCP, make it active, and move to the same target
        robot.add_tcp(name="gripper_tcp",
                      transform=[0.0, 0.0, 0.05, 0.0, 0.0, 0.0],
                      set_active=True)
        robot.set_cartesian_pose(target_tcp_pose)
        logger.success(f"arrived: {robot.get_cartesian_pose()}")

    finally:
        sub.delete()
        robot.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Move 2 cm in Z with the default and a custom TCP, with Rerun."
    )
    parser.add_argument("--ip", type=str, default="192.168.1.100",
                        help="UR controller IP address (default: 192.168.1.100)")
    args = parser.parse_args()

    main(ip=args.ip)
