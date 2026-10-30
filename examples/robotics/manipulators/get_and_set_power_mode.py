"""
Reads the Epson controller's servo power mode, then switches it to low power.

Supports Epson.

Usage:
    python get_and_set_power_mode.py [--ip <ROBOT_IP>]
"""

import argparse

from loguru import logger

from telekinesis.synapse.robots.manipulators import epson


def main(ip: str | None) -> None:
    """Read the Epson power mode and set it to low power."""

    #===================== Create Robot ==========================================
    robot = epson.EpsonCX4A601S(name='EpsonCX4A601S')

    try:
        #===================== Connect Robot ==========================================
        if ip:
            robot.connect(ip=ip)

        # ==================== Run Skill ============================================
        logger.info(f"Current power mode: {robot.power_mode}")

        # Low power limits speed and servo stiffness. Select HIGH only after
        # confirming the physical cell is safe for full-speed operation.
        robot.set_power_mode(epson.EpsonPowerMode.LOW)
        logger.success(f"Power mode after setting LOW: {robot.power_mode}")
    except (ConnectionError, OSError) as e:
        logger.error(f"Error occurred: {e}")
    finally:
        robot.disconnect()
        robot.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Read and set the Epson power mode Synapse example")
    parser.add_argument("--ip", type=str, default=None,
                         help="Epson controller IP address for real hardware, e.g. 192.168.0.1")
    args = parser.parse_args()

    main(ip=args.ip)
