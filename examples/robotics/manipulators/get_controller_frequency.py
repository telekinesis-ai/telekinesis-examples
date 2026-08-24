"""
Measures the controller's update rate by polling get_timestamp() and computing 1 / mean_step_time.

Supports Universal Robots (UR).

Usage:
    python get_controller_frequency.py [--ip <ROBOT_IP>]
"""

import argparse

from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots


def main(ip: str | None) -> None:
    """Log the measured controller update frequency [Hz]."""

    #===================== Create Robot ==========================================
    robot = universal_robots.UniversalRobotsUR10E(name='UR10e')

    try:
        #===================== Connect Robot ==========================================
        if ip:
            robot.connect(ip=ip)

        # ==================== Run Skill ============================================
        frequency = robot.get_controller_frequency(window_s=0.2)
        logger.success(f"Controller frequency [Hz]: {frequency:.2f}")
    except (ConnectionError, OSError) as e:
        logger.error(f"Error occurred: {e}")
    finally:
        robot.disconnect()
        robot.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Read controller frequency Synapse example")
    parser.add_argument("--ip", type=str, default=None,
                         help="UR robot IP address for real hardware, e.g. 192.168.1.100")
    args = parser.parse_args()

    main(ip=args.ip)
