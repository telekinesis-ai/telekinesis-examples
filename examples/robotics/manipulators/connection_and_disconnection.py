"""
Connects to a UR10e, waits briefly, then cleanly disconnects.

Supports Universal Robots (UR), Epson, and Isaac Sim.

Usage:
    python connection_and_disconnection.py [--ip <ROBOT_IP>] [--prim_path <PRIM_PATH>]
"""

import argparse
import time

from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots


def main(ip: str | None, prim_path: str | None) -> None:
    """Connect to a UR10e over real hardware or Isaac Sim, then cleanly disconnect."""

    #===================== Create Robot ==========================================
    robot = universal_robots.UniversalRobotsUR10E(name='UR10e')

    try:
        # ==================== Run Skill ============================================
        if ip:
            robot.connect(ip=ip)
            logger.success(f"Connected to UR10e at {ip}.")
        elif prim_path:
            robot.connect(simulation_prim_path=prim_path)
            logger.success(f"Connected to UR10e at {prim_path}.")

        time.sleep(2)
    except (ConnectionError, OSError) as e:
        logger.error(f"Error occurred: {e}")
    finally:
        robot.disconnect()
        robot.shutdown()
        logger.success("Disconnected.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Connection Synapse example")
    parser.add_argument("--ip", type=str, default=None,
                         help="UR robot IP address for real hardware, e.g. 192.168.1.100")
    parser.add_argument("--prim_path", type=str, default=None,
                         help='Isaac Sim articulation prim path, e.g. "/World/ur10e"')
    args = parser.parse_args()

    main(ip=args.ip, prim_path=args.prim_path)
