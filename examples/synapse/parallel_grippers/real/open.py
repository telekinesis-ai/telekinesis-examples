"""
Demonstrates opening a parallel gripper.

Usage:
    python open.py --ip <GRIPPER_IP>
    python open.py --protocol MODBUS_RTU --serial-port COM4
"""

import argparse
from loguru import logger

from telekinesis.synapse.tools.parallel_grippers import robotiq


def main(ip: str | None, serial_port: str, protocol: str) -> None:
    """Opens a Robotiq gripper fully at 100% speed and 50% force."""

    #===================== Create Gripper ======================================
    gripper = robotiq.Robotiq2F85()

    # ==================== Run Skill ===========================================
    try:
        gripper.connect(ip=ip, serial_port=serial_port, protocol=protocol)
        status = gripper.open(speed=100.0,
                              force=50.0,
                              asynchronous=False)
        logger.success(f"open() status: {status}, "
                       f"position: {gripper.get_current_position():.2f}")
    except (ConnectionError, OSError) as e:
        logger.error(f"Error occurred: {e}")
    finally:
        gripper.disconnect()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Robotiq gripper open")
    p.add_argument("--protocol",
                   choices=["URCAP", "MODBUS_RTU"],
                   default="URCAP")
    p.add_argument("--ip", default=None, help="IP for Robotiq Gripper")
    p.add_argument("--serial-port", dest="serial_port", default="COM4",
                   help="Serial port for MODBUS_RTU")
    args = p.parse_args()

    main(ip=args.ip,
         serial_port=args.serial_port,
         protocol=args.protocol)
