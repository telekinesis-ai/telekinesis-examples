"""
Subscribe to a named robot's state and TF topics and visualize them with Rerun.

Creating a robot with a name starts a RobotStateAndTFPublisher immediately, so
the robot publishes its commanded state and TF transforms at 30 Hz even without
connecting to hardware. This example:

  1. Creates a robot with a name (auto-starts the publisher).
  2. Subscribes to the state topic and logs every field to Rerun as time series.
  3. Subscribes to the TF topic, builds a TransformTree, and visualizes the
     frames live in Rerun via tree.visualize_rerun().

Note on the Rerun timeline: while running offline (no connection), the robot's
state.timestamp is the commanded-cache update time, which is constant between
set_* commands. Logging against it would stack every sample on one time point
and nothing would appear. We therefore drive the timeline with the message
receive time (time.time()), which advances both offline and on a real robot.

This example runs fully offline on the robot's commanded-cache state; no
hardware connection is made.

Install:
    pip install rerun-sdk

Usage:
    python state_and_tf_subscriber_offline.py
"""

import time
from functools import partial

import numpy as np
import rerun as rr
from loguru import logger

from babyros import node

from telekinesis.tf import tftree
from telekinesis.synapse.robots.manipulators import universal_robots


def initialize_rerun(fields: list[str], joint_names: list[str]) -> None:
    """Spawn the Rerun viewer and declare a line style per field once so Rerun
    draws connected lines per joint/component. Only the fields the robot
    actually publishes are declared."""

    rr.init("telekinesis_synapse_state_and_tf_subscriber", spawn=True)

    # Per-component legend labels: joint-space fields use the robot's joint
    # names, TCP wrench uses force/torque axes, other Cartesian fields use pose axes.
    for field in fields:
        if field == "tcp_force":
            names = ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"]
        elif "tcp" in field:
            names = ["x", "y", "z", "rx", "ry", "rz"]
        else:
            names = joint_names
        rr.log(field, rr.SeriesLines(names=names), static=True)


def on_state(msg: dict, fields: list[str]) -> None:
    """node.Subscriber callback — logs every state field to Rerun.

    ``fields`` is bound in main() via functools.partial so the callback keeps
    babyros's single-argument signature.
    """
    # Drive the timeline with the receive time so values advance offline too
    # (the commanded-cache timestamp is constant between set_* commands).
    rr.set_time("log_time", timestamp=time.time())

    for field in fields:
        if field in msg:
            rr.log(field, rr.Scalars(msg[field]))


# The TF tree, built once on the first message and reused after.
tf_tree: tftree.TransformTree | None = None


def on_tf(msg: dict) -> None:
    """node.Subscriber callback — visualizes the robot's TF frames in Rerun.

    Builds the TransformTree on the first message, then just updates each
    frame's transform on later messages (adding any new frame that appears).
    """

    # Create tf_tree
    global tf_tree

    rr.set_time("log_time", timestamp=time.time())

    if tf_tree is None:
        tf_tree = tftree.TransformTree("world")

    # Every transform in the message is relative to the robot base, which the
    # publisher reports with identity. Adding all frames under "world" therefore
    # places the robot base link coincident with the world root.
    for name, transform in msg.items():
        trans = np.array(transform)
        if tf_tree.find_nodes([name]):
            tf_tree.update(name, trans, rot_type="mat")
        else:
            tf_tree.add("world", name, trans, rot_type="mat")

    tf_tree.visualize_rerun(axis_len=0.05)


def main() -> None:
    """Subscribe to a robot's state and TF topics and visualize them in Rerun."""

    try:

        # Create a robot with a name: this auto-starts the state and TF publisher.
        robot = universal_robots.UniversalRobotsUR10E(name="manipulator1")

        # Initialize rerun with robot state fields and joint names
        state_fields = [field for field in robot.get_state() if field != "timestamp"]
        initialize_rerun(fields=state_fields,
                         joint_names=robot.joint_names)

        # Launch subscribers for state and tf
        state_subscriber = node.Subscriber(
            topic=robot.state_publisher_topic, 
            callback=partial(on_state, fields=state_fields)
        )
        tf_subscriber = node.Subscriber(
            topic=robot.tf_publisher_topic,
            callback=on_tf
        )

        # Log the commanded-cache state and TF transforms (offline) for 10 seconds
        time.sleep(10)

    except KeyboardInterrupt:
        logger.info("Interrupted.")

    finally:
        # Clean up subscribers and robot nodes
        logger.info("Shutting down.")
        state_subscriber.delete()
        tf_subscriber.delete()


if __name__ == "__main__":
    main()
