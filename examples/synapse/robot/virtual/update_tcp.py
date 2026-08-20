"""
Example: Demonstrates adding and updating a TCP — offline.

Demonstrates:
- add_tcp()       — register a custom TCP frame
- update_tcp()    — update a custom TCP frame

This example runs offline on the commanded-cache state; no hardware
connection is made.

Usage:
    python update_tcp.py
"""

from loguru import logger

from telekinesis.synapse.robots.manipulators.universal_robots import UniversalRobotsUR10E


def main():
    """Add a TCP, then update its transform, observing it at each step."""

    # Create a UniversalRobotsUR10E instance (no hardware connection)
    robot = UniversalRobotsUR10E()

    # Add new tcp
    new_tcp_pose_in_default_tcp_frame = [0.0, 0.0, 0.1, 0.0, 0.0, 0.0]  # 100 mm along Z-axis
    robot.add_tcp(name="new_tool",
                  transform=new_tcp_pose_in_default_tcp_frame,
                  set_active=True)

    # Active TCP, transform w.r.t default tcp, and TCP pose
    logger.info(f"Active TCP after add_tcp(): {robot.active_tcp}"
                f" \nActive TCP transform: {robot.get_active_tcp_transform()}"
                f" \n TCP pose: {robot.get_cartesian_pose()}")

    # Update the TCP
    updated_tcp_pose_in_default_tcp_frame = [0.0, 0.0, 0.2, 0.0, 0.0, 0.0]  # 200 mm along Z-axis
    robot.update_tcp(name="new_tool",
                     transform=updated_tcp_pose_in_default_tcp_frame)

    # Active TCP, transform w.r.t default tcp, and TCP pose
    logger.info(f"Active TCP after update_tcp(): {robot.active_tcp}"
                f" \nActive TCP transform: {robot.get_active_tcp_transform()}"
                f" \n TCP pose: {robot.get_cartesian_pose()}")


if __name__ == "__main__":
    main()
