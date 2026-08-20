"""
Detach tool example for the Synapse SDK.

``detach_tool`` clears the Rerun visualization of a previously attached
tool. A tool assembled onto the arm in Isaac Sim is not affected — the
simulation offers no way to take an assembled tool back off; this call
only affects the visualization.

Supported for all robots offline, and both Universal Robots (UR) and Epson
in real.

Usage:
    python detach_tool.py [--ip <ROBOT_IP>]
"""

import argparse

from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots
from telekinesis.synapse.tools.parallel_grippers import onrobot


def main(ip: str):
    """Attach a gripper, visualize it, then detach it again."""

    #===================== Create Robot and Gripper =============================
    robot = universal_robots.UniversalRobotsUR10E()
    gripper = onrobot.OnRobotRG6()
    robot.connect(ip=ip)

    # ==================== Run Skill ============================================
    try:
        robot.attach_tool(gripper)
        robot.visualize_rerun(live=False)
        logger.info("Gripper attached and visualized.")

        robot.detach_tool()
        logger.success("Gripper detached from visualization.")
    finally:
        robot.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detach a tool from the robot")
    parser.add_argument("--ip", type=str, default="192.168.1.100", help="UR robot IP address (default: 192.168.1.100)")
    args = parser.parse_args()

    main(ip=args.ip)
