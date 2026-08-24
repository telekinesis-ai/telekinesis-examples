"""
Attaches a gripper and visualizes it, then detaches it from the Rerun visualization again.

Supports Universal Robots (UR) and Epson.

Usage:
    python detach_tool.py [--ip <ROBOT_IP>]
"""

import argparse

from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots
from telekinesis.synapse.tools.parallel_grippers import onrobot


def main(ip: str) -> None:
    """Attach a gripper, visualize it, then detach it again."""

    #===================== Create Robot and Gripper =============================
    robot = universal_robots.UniversalRobotsUR10E(name='UR10e')
    gripper = onrobot.OnRobotRG6()

    try:
        #===================== Connect Robot ==========================================
        robot.connect(ip=ip)

        # ==================== Run Skill ============================================
        robot.attach_tool(gripper)
        robot.visualize_rerun(live=False)
        logger.info("Gripper attached and visualized.")

        robot.detach_tool()
        logger.success("Gripper detached from visualization.")
    except (ConnectionError, OSError) as e:
        logger.error(f"Error occurred: {e}")
    finally:
        robot.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detach a tool from the robot")
    parser.add_argument("--ip", type=str, default="192.168.1.100", help="UR robot IP address (default: 192.168.1.100)")
    args = parser.parse_args()

    main(ip=args.ip)
