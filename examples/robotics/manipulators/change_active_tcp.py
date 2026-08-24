"""
Registers several TCP frames and switches the active one.

Supports Universal Robots (UR), Epson, virtual, and Isaac Sim.

Usage:
    python change_active_tcp.py [--ip <ROBOT_IP>] [--prim_path <PRIM_PATH>]
"""

import argparse

from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots


def main(ip: str | None, prim_path: str | None) -> None:
    """Change the active TCP and observe it before and after each change."""

    #===================== Create Robot ==========================================
    robot = universal_robots.UniversalRobotsUR10E(name='UR10e')

    # ==================== Visualization (Optional) =============================
    robot.visualize_rerun(live=True)

    try:
        #===================== Connect Robot ==========================================
        if ip:
            robot.connect(ip=ip)
        elif prim_path:
            robot.connect(simulation_prim_path=prim_path)

        # ==================== Run Skill ============================================
        robot.add_tcp(name="camera_tip",
                      transform=[0.0, 0.0, 0.1, 0.0, 0.0, 0.0],  # 100 mm along Z-axis
                      set_active=True)
        robot.add_tcp(name="gripper_tip",
                      transform=[0.0, 0.0, 0.2, 0.0, 0.0, 0.0],
                      set_active=False)
        robot.add_tcp(name="laser_tip",
                      transform=[0.0, 0.0, 0.3, 0.0, 0.0, 0.0],
                      set_active=False)

        logger.info(f"Active TCP after add_tcp(): {robot.active_tcp}"
                    f" \nActive TCP transform: {robot.get_active_tcp_transform()}"
                    f" \n TCP pose: {robot.get_cartesian_pose()}")

        robot.active_tcp = "gripper_tip"

        logger.info(f"Active TCP after changing active TCP: {robot.active_tcp}"
                    f" \nActive TCP transform: {robot.get_active_tcp_transform()}"
                    f" \n TCP pose: {robot.get_cartesian_pose()}")

        robot.active_tcp = "laser_tip"

        logger.info(f"Active TCP after changing active TCP again: {robot.active_tcp}"
                    f" \nActive TCP transform: {robot.get_active_tcp_transform()}"
                    f" \n TCP pose: {robot.get_cartesian_pose()}")
    except (ConnectionError, OSError) as e:
        logger.error(f"Error occurred: {e}")
    finally:
        robot.disconnect()
        robot.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Register several TCPs on the robot and switch the active one.")
    parser.add_argument("--ip", type=str, default=None,
                         help="UR robot IP address for real hardware, e.g. 192.168.1.100")
    parser.add_argument("--prim_path", type=str, default=None,
                         help='Isaac Sim articulation prim path, e.g. "/World/ur10e"')
    args = parser.parse_args()

    main(ip=args.ip, prim_path=args.prim_path)
