"""
Registers a custom TCP frame and inspects the active TCP before and after.

Supports Universal Robots (UR), Epson, virtual, and Isaac Sim.

Usage:
    python add_tcp.py [--ip <ROBOT_IP>] [--prim_path <PRIM_PATH>]
"""

import argparse

from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots


def main(ip: str | None, prim_path: str | None) -> None:
    """Observe the active TCP and its transform before and after add_tcp()."""

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
            robot.set_joint_positions(robot.default_joint_configuration)
            
        # ==================== Run Skill ============================================
        logger.info(f"Active TCP before add_tcp(): {robot.active_tcp}"
                    f" \nActive TCP transform: {robot.get_active_tcp_transform()}"
                    f" \n TCP pose: {robot.get_cartesian_pose()}")

        new_tcp_pose_in_default_tcp_frame = [0.0, 0.0, 0.1, 0.0, 0.0, 0.0]  # 100 mm along Z-axis
        robot.add_tcp(name="new_tool",
                      transform=new_tcp_pose_in_default_tcp_frame,
                      set_active=True)

        logger.info(f"Active TCP after add_tcp(): {robot.active_tcp}"
                    f" \nActive TCP transform: {robot.get_active_tcp_transform()}"
                    f" \n TCP pose: {robot.get_cartesian_pose()}")
    except (ConnectionError, OSError) as e:
        logger.error(f"Error occurred: {e}")
    finally:
        robot.disconnect()
        robot.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add a custom TCP to the robot and inspect the active TCP.")
    parser.add_argument("--ip", type=str, default=None,
                         help="UR robot IP address for real hardware, e.g. 192.168.1.100")
    parser.add_argument("--prim_path", type=str, default=None,
                         help='Isaac Sim articulation prim path, e.g. "/World/ur10e"')
    args = parser.parse_args()

    main(ip=args.ip, prim_path=args.prim_path)
