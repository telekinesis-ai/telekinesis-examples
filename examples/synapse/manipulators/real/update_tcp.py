"""
Example: Demonstrates adding and updating TCP

Demonstrates:
- add_tcp()                   — register a custom TCP frame and push it to the controller
- update_tcp()                — update a custom TCP frame in the controller

Currently supported only for real hardware. Works on Universal Robots (UR) and Epson.

For an offline version, refer to tcp/offline/update_tcp.py

Usage:
    python update_tcp.py [--ip <ROBOT_IP>]
"""

import argparse
from loguru import logger

from telekinesis.synapse.robots.manipulators.universal_robots import UniversalRobotsUR10E

def main():
    """
    Add and update TCP
    """

    # Parse command-line arguments for the UR controller IP address
    parser = argparse.ArgumentParser(description="Add and then update a custom TCP on a real UR10E.")
    parser.add_argument("--ip", type=str, default="192.168.1.100", help="UR controller IP address (default: 192.168.1.100)")
    args = parser.parse_args()

    #===================== Create Robot ==========================================
    robot = UniversalRobotsUR10E()
    robot.connect(ip=args.ip)

    # ==================== Run Skill ============================================
    try:
        new_tcp_pose_in_default_tcp_frame = [0.0, 0.0, 0.1, 0.0, 0.0, 0.0]  # 100 mm along Z-axis
        robot.add_tcp(name="new_tool",
                      transform=new_tcp_pose_in_default_tcp_frame,
                      set_active=True)

        # Get updated Active TCP, transform w.r.t default tcp, and TCP pose
        logger.info(f"Active TCP after add_tcp(): {robot.active_tcp}"
                    f" \nActive TCP transform: {robot.get_active_tcp_transform()}"
                    f" \n TCP pose: {robot.get_cartesian_pose()}")

        # Update the TCP
        updated_tcp_pose_in_default_tcp_frame = [0.0, 0.0, 0.2, 0.0, 0.0, 0.0]  # 200 mm along Z-axis
        robot.update_tcp(name="new_tool",
                         transform=updated_tcp_pose_in_default_tcp_frame)

        # Get updated Active TCP, transform w.r.t default tcp, and TCP pose
        logger.info(f"Active TCP after update_tcp(): {robot.active_tcp}"
                    f" \nActive TCP transform: {robot.get_active_tcp_transform()}"
                    f" \n TCP pose: {robot.get_cartesian_pose()}")

        # ==================== Visualization (Optional) =========================
        robot.visualize_rerun(live=False)

    finally:
        # Disconnect
        robot.disconnect()


if __name__ == "__main__":
    main()
