"""
Demonstrates setting the maximum vacuum-pump speed of a suction gripper.

Only supported on the URCAP protocol: the pump's serial protocol documents no
parameter index for the maximum pump PWM, so on MODBUS_RTU it can only be set
on the pump display or through the URCap.

Usage:
    python set_max_pump_speed.py --ip <ROBOT_IP>
"""

import argparse
from loguru import logger

from telekinesis.synapse.tools.suction_grippers import piab


def main(ip: str | None, serial_port: str, protocol: str) -> None:
    """Sets the maximum pump speed of a Piab gripper to 80%."""

    #===================== Create Gripper ======================================
    gripper = piab.PiabPiCobotElectric()

    # ==================== Run Skill ===========================================
    try:
        gripper.connect(ip=ip, serial_port=serial_port, protocol=protocol)
        gripper.set_max_pump_speed(max_speed=80)
        logger.success("Maximum pump speed set to 80%.")
    except (ConnectionError, OSError) as e:
        logger.error(f"Error occurred: {e}")
    except NotImplementedError as e:
        logger.warning(f"Not supported on protocol '{protocol}': {e}")
    finally:
        gripper.disconnect()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Piab gripper set max pump speed")
    p.add_argument("--protocol",
                   choices=["URCAP", "MODBUS_RTU"],
                   default="URCAP")
    p.add_argument("--ip", default="192.168.2.2", help="IP for Robot Controller")
    p.add_argument("--serial-port", dest="serial_port", default="COM3",
                   help="Serial port for MODBUS_RTU")
    args = p.parse_args()

    main(ip=args.ip,
         serial_port=args.serial_port,
         protocol=args.protocol)
