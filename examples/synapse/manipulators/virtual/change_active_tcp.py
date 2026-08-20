"""
Example: Demonstrates how to change the active TCP — offline.

Demonstrates:
- add_tcp()      — register a custom TCP frame
- active_tcp     — change which frame is currently active

This example runs offline on the commanded-cache state; no hardware
connection is made.

Usage:
    python change_active_tcp.py
"""

from loguru import logger

from telekinesis.synapse.robots.manipulators.universal_robots import UniversalRobotsUR10E


def main():
    """Change the active TCP and observe it before and after each change."""

    # Create a UniversalRobotsUR10E instance (no hardware connection)
    robot = UniversalRobotsUR10E()

    # Register a few custom TCP frames
    robot.add_tcp(name="camera_tip",
                  transform=[0.0, 0.0, 0.1, 0.0, 0.0, 0.0],
                  set_active=True)
    robot.add_tcp(name="gripper_tip",
                  transform=[0.0, 0.0, 0.2, 0.0, 0.0, 0.0],
                  set_active=False)
    robot.add_tcp(name="laser_tip",
                  transform=[0.0, 0.0, 0.3, 0.0, 0.0, 0.0],
                  set_active=False)

    # Active TCP, transform w.r.t default tcp, and TCP pose
    logger.info(f"Active TCP after add_tcp(): {robot.active_tcp}"
                f" \nActive TCP transform: {robot.get_active_tcp_transform()}"
                f" \n TCP pose: {robot.get_cartesian_pose()}")

    # Change the active TCP
    robot.active_tcp = "gripper_tip"
    logger.info(f"Active TCP after changing active TCP: {robot.active_tcp}"
                f" \nActive TCP transform: {robot.get_active_tcp_transform()}"
                f" \n TCP pose: {robot.get_cartesian_pose()}")

    # Change the active TCP again
    robot.active_tcp = "laser_tip"
    logger.info(f"Active TCP after changing active TCP again: {robot.active_tcp}"
                f" \nActive TCP transform: {robot.get_active_tcp_transform()}"
                f" \n TCP pose: {robot.get_cartesian_pose()}")


if __name__ == "__main__":
    main()
