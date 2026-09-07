"""
Recovers a Robot controller after automatic motor-on was rejected.

Supports Universal Robots (UR) and Epson.

Usage:
    python reset_controller.py [--ip <ROBOT_IP>]
"""

import argparse

from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots


def main(ip: str | None) -> None:
    """Reset Epson and turn motors on after the physical cause is cleared."""

    #===================== Create Robot ==========================================
    robot = universal_robots.UniversalRobotsUR10E(name='UR10e')

    try:
        #===================== Connect Robot ==========================================
        if ip:
            robot.connect(ip=ip)

        # ==================== Run Skill ============================================
        status = robot.get_status()
        if status.controller is not None and not status.controller.motors_on:
            logger.warning(
                "Motors are off. Release any physical E-stop or safeguard before "
                "continuing with controller reset."
            )
            robot.reset_controller()
            logger.success("Controller reset and motors turned on.")
        else:
            logger.success("Controller connected with motors already on; no reset required.")
    except (ConnectionError, OSError) as e:
        logger.error(f"Error occurred: {e}")
    finally:
        robot.disconnect()
        robot.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reset an Epson controller Synapse example")
    parser.add_argument("--ip", type=str, default=None,
                         help="Epson controller IP address for real hardware, e.g. 192.168.0.1")
    args = parser.parse_args()

    main(ip=args.ip)
