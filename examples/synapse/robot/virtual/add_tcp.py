"""
Example: Add a custom TCP to the robot — offline.

Demonstrates:
- add_tcp()                   — register a custom TCP frame
- get_active_tcp_transform()  — read the active TCP offset (metres, Euler-XYZ degrees)
- active_tcp                  — check which frame is currently active

This example runs offline on the commanded-cache state; no hardware
connection is made.

Usage:
    python add_tcp.py
"""

from loguru import logger

from telekinesis.synapse.robots.manipulators.universal_robots import UniversalRobotsUR10E


def main():
    """Observe the active TCP and its transform before and after add_tcp()."""

    # Create a UniversalRobotsUR10E instance (no hardware connection)
    robot = UniversalRobotsUR10E()

    # Current Active TCP, transform w.r.t default tcp, and current TCP pose
    logger.info(f"Active TCP before add_tcp(): {robot.active_tcp}"
                f" \nActive TCP transform: {robot.get_active_tcp_transform()}"
                f" \n TCP pose: {robot.get_cartesian_pose()}")

    # Add new tcp
    new_tcp_pose_in_default_tcp_frame = [0.0, 0.0, 0.1, 0.0, 0.0, 0.0]  # 100 mm along Z-axis
    robot.add_tcp(name="new_tool",
                  transform=new_tcp_pose_in_default_tcp_frame,
                  set_active=True)

    # Get updated Active TCP, transform w.r.t default tcp, and TCP pose
    logger.info(f"Active TCP after add_tcp(): {robot.active_tcp}"
                f" \nActive TCP transform: {robot.get_active_tcp_transform()}"
                f" \n TCP pose: {robot.get_cartesian_pose()}")


if __name__ == "__main__":
    main()
