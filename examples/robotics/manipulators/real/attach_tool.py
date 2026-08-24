"""
Attaches an OnRobot RG6 gripper to a UR10e, registers its TCP, and visualizes it in Rerun.

Supports Universal Robots (UR), Epson, and virtual.

Usage:
    python attach_tool.py [--ip <ROBOT_IP>]
"""

import argparse

from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots
from telekinesis.synapse.tools.parallel_grippers import onrobot


def main(ip: str) -> None:
    """Attach an OnRobot RG6 gripper to a UR10e and visualize in Rerun."""

    #===================== Create Robot and Gripper =============================
    robot = universal_robots.UniversalRobotsUR10E(name='UR10e')
    gripper = onrobot.OnRobotRG6()

    try:
        #===================== Connect Robot ==========================================
        robot.connect(ip=ip)

        # ==================== Run Skill ============================================
        robot.attach_tool(gripper)
        robot.add_tcp(name="gripper_tip",
                      transform=[0.0, 0.0, 0.18, 0.0, 0.0, 0.0],
                      set_active=True)

        # ==================== Visualization (Optional) =============================
        robot.visualize_rerun(live=False)
    except (ConnectionError, OSError) as e:
        logger.error(f"Error occurred: {e}")
    finally:
        robot.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Attach a gripper to a real UR10e and visualize it")
    parser.add_argument("--ip", type=str, default="192.168.1.100", help="UR robot IP address (default: 192.168.1.100)")
    args = parser.parse_args()

    main(ip=args.ip)
