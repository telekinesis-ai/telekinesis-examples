"""
Reads the properties of a manipulator, sets its publish rate, and shows the robot in Rerun.

Supports all manipulators. No hardware is needed: these properties come from the model.

Usage:
    python robot_properties.py
    python robot_properties.py --save-rrd robot_properties.rrd
"""

import argparse

import rerun as rr
from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots


def init_rerun(app_id: str, save_rrd: str | None) -> "rr.RecordingStream":
    """Start a Rerun recording, live in the viewer or saved to an .rrd file."""
    rr.init(app_id, spawn=save_rrd is None)
    recording = rr.get_global_data_recording()
    if save_rrd:
        recording.save(save_rrd)
    return recording


def main(save_rrd: str | None) -> None:
    """Log the main properties of a UR10e."""

    #===================== Start Rerun ==========================================
    recording = init_rerun("telekinesis_synapse_robot_properties", save_rrd)

    #===================== Create Robot ==========================================
    # name= starts the state/TF publisher, which enables the topic properties.
    robot = universal_robots.UniversalRobotsUR10E(name="ur10e")

    try:
        # ==================== Run Skill ============================================
        # Only some models define their joint velocity limits. The UR10e does not,
        # so reading it raises NotImplementedError.
        try:
            max_joint_velocities = f"{robot.max_joint_velocities} deg/s"
        except NotImplementedError:
            max_joint_velocities = "not defined for this model"

        properties = {
            "joint_names": robot.joint_names,
            "frame_names": f"{len(robot.frame_names)} frames",
            "custom_tcps": robot.custom_tcps,
            "max_joint_velocities": max_joint_velocities,
            "state_publisher_topic": robot.state_publisher_topic,
            "tf_publisher_topic": robot.tf_publisher_topic,
            "control_frequency": f"{robot.control_frequency} Hz",
        }
        for name, value in properties.items():
            logger.info(f"{name}: {value}")

        robot.publish_frequency = 30.0
        logger.success(f"publish_frequency: {robot.publish_frequency} Hz")
        properties["publish_frequency"] = f"{robot.publish_frequency} Hz"

        # ==================== Visualization (Optional) =============================
        # The robot and its TCP frames are drawn once, and every property is logged
        # as a text line next to the 3D view.
        robot.visualize_rerun(recording_stream=recording, live=False)
        for name, value in properties.items():
            rr.log(f"/properties/{name}", rr.TextLog(f"{name}: {value}"))
    finally:
        robot.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Robot properties Synapse example")
    parser.add_argument("--save-rrd", type=str, default=None,
                         help="Save the Rerun recording to this .rrd file instead of opening the viewer")
    args = parser.parse_args()

    main(save_rrd=args.save_rrd)
