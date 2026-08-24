"""
Logs whether the manipulator state is being driven by live hardware, before and after connecting.

Supports Universal Robots (UR), Epson, virtual, and Isaac Sim.

Usage:
    python is_connected.py [--ip <ROBOT_IP>] [--prim_path <PRIM_PATH>]
"""

import argparse

from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots


def main(ip: str | None, prim_path: str | None) -> None:
    """Log is_connected before and after connecting."""

    #===================== Create Robot ==========================================
    robot = universal_robots.UniversalRobotsUR10E(name='UR10e')
    logger.info(f"is_connected before connect(): {robot.is_connected()}")

    try:
        #===================== Connect Robot ==========================================
        if ip:
            robot.connect(ip=ip)
        elif prim_path:
            robot.connect(simulation_prim_path=prim_path)

        # ==================== Run Skill ============================================
        logger.success(f"is_connected while connected: {robot.is_connected()}")
    except (ConnectionError, OSError) as e:
        logger.error(f"Error occurred: {e}")
    finally:
        robot.disconnect()
        robot.shutdown()
    logger.info(f"is_connected after disconnect(): {robot.is_connected()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check connection status Synapse example")
    parser.add_argument("--ip", type=str, default=None,
                         help="UR robot IP address for real hardware, e.g. 192.168.1.100")
    parser.add_argument("--prim_path", type=str, default=None,
                         help='Isaac Sim articulation prim path, e.g. "/World/ur10e"')
    args = parser.parse_args()

    main(ip=args.ip, prim_path=args.prim_path)
