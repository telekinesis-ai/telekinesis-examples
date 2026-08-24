"""
Attaches an OnRobot RG6 gripper to a UR10e, registers its TCP, and visualizes it in Rerun.

Supports Universal Robots (UR), Epson, virtual, and Isaac Sim.

Usage:
    python attach_tool.py [--ip <ROBOT_IP>] [--prim_path <PRIM_PATH>]
"""

import argparse

from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots
from telekinesis.synapse.tools.parallel_grippers import onrobot


def main(ip: str | None, prim_path: str | None) -> None:
    """Attach an OnRobot RG6 gripper to a UR10e and visualize in Rerun."""

    #===================== Create Robot and Gripper =============================
    robot = universal_robots.UniversalRobotsUR10E(name='UR10e')
    gripper = onrobot.OnRobotRG6()

    # ==================== Visualization (Optional) =============================
    robot.visualize_rerun(live=True)

    try:
        #===================== Connect Robot ==========================================
        if ip:
            robot.connect(ip=ip)
        elif prim_path:
            robot.connect(simulation_prim_path=prim_path)

        # ==================== Run Skill ============================================
        robot.attach_tool(gripper)
        robot.add_tcp(name="gripper_tip",
                      transform=[0.0, 0.0, 0.18, 0.0, 0.0, 0.0],
                      set_active=True)
    except (ConnectionError, OSError) as e:
        logger.error(f"Error occurred: {e}")
    finally:
        robot.disconnect()
        robot.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Attach a gripper to the robot and visualize it")
    parser.add_argument("--ip", type=str, default=None,
                         help="UR robot IP address for real hardware, e.g. 192.168.1.100")
    parser.add_argument("--prim_path", type=str, default=None,
                         help='Isaac Sim articulation prim path, e.g. "/World/ur10e"')
    args = parser.parse_args()

    main(ip=args.ip, prim_path=args.prim_path)
