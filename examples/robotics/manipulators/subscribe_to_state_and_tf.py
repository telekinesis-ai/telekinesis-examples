"""
Subscribes to a named robot's state and TF topics and plots what arrives in Rerun while the robot moves.

The robot runs offline here, so no hardware or simulator is needed. Pass --ip or --prim_path
to follow a real robot or an Isaac Sim articulation instead.

Supports Universal Robots (UR), Epson, virtual, and Isaac Sim.

Usage:
    python subscribe_to_state_and_tf.py [--ip <ROBOT_IP>] [--prim_path <PRIM_PATH>]
    python subscribe_to_state_and_tf.py --save-rrd subscribe_to_state_and_tf.rrd
"""

import argparse
import time

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from babyros import node
from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots

TCP_AXIS_LABELS = ["x", "y", "z", "rx", "ry", "rz"]
TCP_AXIS_COLORS = [
    [31, 119, 180],
    [255, 127, 14],
    [44, 160, 44],
    [214, 39, 40],
    [148, 103, 189],
    [140, 86, 75],
]
JOINT_COLOR = [31, 119, 180]

# Width of the plotted lines, in UI points; wider lines are easier to read.
LINE_WIDTH = 3.0

# Joint angles in degrees: the home pose, then a pose with the base rotated.
HOME_JOINTS = [0.0, -90.0, 90.0, -90.0, -90.0, 0.0]
ROTATED_JOINTS = [-45.0, -90.0, 90.0, -90.0, -90.0, 0.0]

# Number of messages received on each topic.
received = {"state": 0, "tf": 0}


def init_rerun(app_id: str, save_rrd: str | None) -> "rr.RecordingStream":
    """Start a Rerun recording, live in the viewer or saved to an .rrd file."""
    rr.init(app_id, spawn=save_rrd is None)
    recording = rr.get_global_data_recording()
    if save_rrd:
        recording.save(save_rrd)
    return recording


def configure_series(path: str, name: str, color: list[int]) -> None:
    """Set the name, color, and width of a scalar plot line."""
    rr.log(path, rr.SeriesLines(names=[name], colors=[color], widths=[LINE_WIDTH]), static=True)


def on_state(msg: dict) -> None:
    """Log one state message: per-joint position and TCP pose, at its own timestamp."""
    received["state"] += 1
    rr.set_time("time", timestamp=msg["timestamp"])
    for joint_index, position in enumerate(msg["joint_positions"]):
        rr.log(f"joint_{joint_index}/position", rr.Scalars(float(position)))
    for axis_index, label in enumerate(TCP_AXIS_LABELS):
        rr.log(f"tcp_pose/{label}", rr.Scalars(float(msg["tcp_pose"][axis_index])))


def on_tf(msg: dict) -> None:
    """Log one TF message: every published frame as a Rerun Transform3D."""
    received["tf"] += 1
    rr.set_time("time", timestamp=time.time())
    for frame_name, matrix in msg.items():
        transform = np.asarray(matrix, dtype=float)
        rr.log(f"tf/{frame_name}",
               rr.Transform3D(translation=transform[:3, 3], mat3x3=transform[:3, :3]))


def main(ip: str | None, prim_path: str | None, save_rrd: str | None) -> None:
    """Move a robot and plot the state and TF it publishes."""

    #===================== Start Rerun ==========================================
    init_rerun("telekinesis_synapse_subscribe_to_state_and_tf", save_rrd)

    #===================== Create Robot ==========================================
    # The name is required: without it the robot creates no publisher.
    robot = universal_robots.UniversalRobotsUR10E(name="ur10e")

    # ==================== Visualization (Optional) =============================
    # The TF tree in a 3D view, next to a grid of plots for the joint angles. An
    # explicit layout keeps every logged entity visible.
    blueprint = rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(origin="tf", name="TF frames"),
            rrb.Grid(
                *(
                    rrb.TimeSeriesView(origin=f"joint_{joint_index}/position",
                                       name=f"joint_{joint_index} [deg]")
                    for joint_index in range(robot.ndof)
                ),
                grid_columns=3,
            ),
            column_shares=[1, 2],
        )
    )
    rr.send_blueprint(blueprint, make_active=True, make_default=True)
    for joint_index in range(robot.ndof):
        configure_series(f"joint_{joint_index}/position", f"joint_{joint_index} [deg]", JOINT_COLOR)
    for label, color in zip(TCP_AXIS_LABELS, TCP_AXIS_COLORS):
        configure_series(f"tcp_pose/{label}", label, color)

    #===================== Subscribe =============================================
    logger.info(f"State topic: {robot.state_publisher_topic}")
    logger.info(f"TF topic: {robot.tf_publisher_topic}")
    state_subscriber = node.Subscriber(topic=robot.state_publisher_topic, callback=on_state)
    tf_subscriber = node.Subscriber(topic=robot.tf_publisher_topic, callback=on_tf)

    try:
        #===================== Connect Robot ======================================
        if ip:
            robot.connect(ip=ip)
        elif prim_path:
            robot.connect(simulation_prim_path=prim_path)

        # ==================== Run Skill ==========================================
        robot.set_joint_positions(HOME_JOINTS)
        robot.set_joint_positions(ROTATED_JOINTS)
        time.sleep(1.0)

        logger.success(f"State messages received: {received['state']}")
        logger.success(f"TF messages received: {received['tf']}")
        logger.success(f"Measured publish rate: {robot.get_publisher_hz()} Hz")
    except (ConnectionError, OSError) as e:
        logger.error(f"Error occurred: {e}")
    finally:
        state_subscriber.delete()
        tf_subscriber.delete()
        if ip or prim_path:
            robot.disconnect()
        robot.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Subscribe to robot state and TF Synapse example")
    parser.add_argument("--ip", type=str, default=None,
                         help="UR robot IP address for real hardware, e.g. 192.168.1.100")
    parser.add_argument("--prim_path", type=str, default=None,
                         help='Isaac Sim articulation prim path, e.g. "/World/ur10e"')
    parser.add_argument("--save-rrd", type=str, default=None,
                         help="Save the Rerun recording to this .rrd file instead of opening the viewer")
    args = parser.parse_args()

    main(ip=args.ip, prim_path=args.prim_path, save_rrd=args.save_rrd)
