"""
Example: Add a custom TCP to the robot

Usage:
    python add_tcp.py [--ip <ROBOT_IP>]

Demonstrates:
- add_tcp()                   — register a custom TCP frame and push it to the controller
- get_active_tcp_transform()  — read the active TCP offset (metres, Euler-XYZ degrees)
- active_tcp                  — check which frame is currently active

Currently supported only for real hardware. Works on Universal Robots (UR) and Epson.

For an offline version, refer to tcp/offline/add_tcp.py
"""

import argparse
from loguru import logger

from telekinesis.synapse.robots.manipulators.universal_robots import UniversalRobotsUR10E

def main():
    """Observe the active TCP and its transform before and after add_tcp()."""

    # Parse command-line arguments for the UR controller IP address
    parser = argparse.ArgumentParser(description="Add a custom TCP to a real UR10E and inspect the active TCP.")
    parser.add_argument("--ip", type=str, default="192.168.1.100", help="UR controller IP address (default: 192.168.1.100)")
    args = parser.parse_args()

    #===================== Create Robot ==========================================
    robot = UniversalRobotsUR10E()
    robot.connect(ip=args.ip)

    # ==================== Run Skill ============================================
    try:
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

        # ==================== Visualization (Optional) =========================
        robot.visualize_rerun(live=False)

    finally:
        # Disconnect
        robot.disconnect()


if __name__ == "__main__":
    main()
