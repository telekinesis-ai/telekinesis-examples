"""
Reads the active TCP's offset from the robot's default TCP.

Supports Universal Robots (UR), Epson, virtual, and Isaac Sim.

Usage:
    python get_active_tcp_transform.py [--ip <ROBOT_IP>] [--prim_path <PRIM_PATH>]
"""

import argparse

from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots


def main(ip: str | None, prim_path: str | None) -> None:
    """Add a TCP, make it active and read its transform."""

    #===================== Create Robot ==========================================
    robot = universal_robots.UniversalRobotsUR10E(name='UR10e')

    try:
        #===================== Connect Robot ==========================================
        if ip:
            robot.connect(ip=ip)
        elif prim_path:
            robot.connect(simulation_prim_path=prim_path)
            robot.set_joint_positions(robot.default_joint_configuration)

        # ==================== Run Skill ============================================
        logger.info(f"Active TCP: {robot.active_tcp}"
                    f" \nActive TCP transform: {robot.get_active_tcp_transform()}")

        # A TCP 15 cm along the tool Z axis [x, y, z, rx, ry, rz] (m, deg)
        robot.add_tcp(name="tool_tip", transform=[0, 0, 0.15, 0, 0, 0], set_active=True)
        logger.success(f"Active TCP: {robot.active_tcp}"
                       f" \nActive TCP transform: {robot.get_active_tcp_transform()}")
    except (ConnectionError, OSError) as e:
        logger.error(f"Error occurred: {e}")
    finally:
        robot.disconnect()
        robot.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Read the active TCP transform")
    parser.add_argument("--ip", type=str, default=None,
                         help="UR robot IP address for real hardware, e.g. 192.168.1.100")
    parser.add_argument("--prim_path", type=str, default=None,
                         help='Isaac Sim articulation prim path, e.g. "/World/ur10e"')
    args = parser.parse_args()

    main(ip=args.ip, prim_path=args.prim_path)
