"""
Creates a manipulator and a parallel gripper from class-name strings, and shows them in Rerun.

Supports all registered devices. No hardware is needed: the devices are created unconnected.

Usage:
    python robot_registry.py [--robot <CLASS_NAME>] [--gripper <CLASS_NAME>]
    python robot_registry.py --robot UniversalRobotsUR10E --gripper RobotiqHandE --save-rrd robot_registry.rrd
"""

import argparse
from importlib import import_module

import rerun as rr
from loguru import logger

from telekinesis.synapse.robots.manipulators import registry
from telekinesis.synapse.tools.parallel_grippers.registry import PARALLEL_GRIPPERS


def init_rerun(app_id: str, save_rrd: str | None) -> "rr.RecordingStream":
    """Start a Rerun recording, live in the viewer or saved to an .rrd file."""
    rr.init(app_id, spawn=save_rrd is None)
    recording = rr.get_global_data_recording()
    if save_rrd:
        recording.save(save_rrd)
    return recording


def main(robot_class_name: str, gripper_class_name: str, save_rrd: str | None) -> None:
    """Create a robot and a gripper from their registry names."""

    #===================== Look Up Classes ========================================
    if robot_class_name not in registry.MANIPULATORS:
        raise SystemExit(f"Unknown manipulator {robot_class_name!r}. "
                         f"Available: {', '.join(sorted(registry.MANIPULATORS))}")
    if gripper_class_name not in PARALLEL_GRIPPERS:
        raise SystemExit(f"Unknown parallel gripper {gripper_class_name!r}. "
                         f"Available: {', '.join(sorted(PARALLEL_GRIPPERS))}")

    robot_module, robot_class = registry.MANIPULATORS[robot_class_name]
    gripper_module, gripper_class = PARALLEL_GRIPPERS[gripper_class_name]

    #===================== Start Rerun ============================================
    recording = init_rerun("telekinesis_synapse_robot_registry", save_rrd)

    #===================== Create Devices =========================================
    robot = getattr(import_module(robot_module), robot_class)(name="robot")
    gripper = getattr(import_module(gripper_module), gripper_class)()

    try:
        # ==================== Run Skill ================================================
        logger.success(f"Created {type(robot).__name__} and {type(gripper).__name__}")
        logger.info(f"{len(registry.MANIPULATORS)} manipulators are registered")

        # ==================== Visualization (Optional) =============================
        # Draw the robot, then attach the gripper so it is drawn on the flange.
        robot.visualize_rerun(recording_stream=recording, live=False)
        robot.attach_tool(gripper)
        rr.log("/registry/devices",
               rr.TextLog(f"Created {type(robot).__name__} and {type(gripper).__name__} "
                          f"from the registries"))
    finally:
        robot.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Device registry Synapse example")
    parser.add_argument("--robot", type=str, default="UniversalRobotsUR10E",
                        help="Manipulator class name, e.g. UniversalRobotsUR10E")
    parser.add_argument("--gripper", type=str, default="RobotiqHandE",
                        help="Parallel gripper class name, e.g. RobotiqHandE")
    parser.add_argument("--save-rrd", type=str, default=None,
                        help="Save the Rerun recording to this .rrd file instead of opening the viewer")
    args = parser.parse_args()

    main(robot_class_name=args.robot,
         gripper_class_name=args.gripper,
         save_rrd=args.save_rrd)
