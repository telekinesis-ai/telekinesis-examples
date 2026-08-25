"""
Demonstrates setting the vacuum level of a suction gripper.

Supports Piab grippers.

Usage:
    python set_vacuum_level.py --ip <ROBOT_IP>
    python set_vacuum_level.py --protocol MODBUS_RTU --serial-port COM3
    python set_vacuum_level.py --prim_path <PRIM_PATH>
"""

import argparse
from loguru import logger

from telekinesis.synapse.tools.suction_grippers import piab


def main(ip: str | None,
         serial_port: str,
         protocol: str,
         prim_path: str | None) -> None:
    """Sets the vacuum level of a Piab gripper to 60%."""

    #===================== Create Gripper ======================================
    gripper = piab.PiabPiCobotElectric()

    # ==================== Run Skill ===========================================
    try:
        #===================== Connect Gripper =================================
        if prim_path:
            gripper.connect(simulation_prim_path=prim_path)
        else:
            gripper.connect(ip=ip, serial_port=serial_port, protocol=protocol)

        gripper.set_vacuum_level(vacuum_level=60, unit="percentage")
        logger.success(f"Vacuum level set; effective: "
                       f"{gripper.get_vacuum_level(unit='percentage')}%")
    except (ConnectionError, OSError) as e:
        logger.error(f"Error occurred: {e}")
    finally:
        gripper.disconnect()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Piab gripper set vacuum level")
    p.add_argument("--protocol",
                   choices=["URCAP", "MODBUS_RTU"],
                   default="URCAP")
    p.add_argument("--ip", default="192.168.2.2", help="IP for Robot Controller")
    p.add_argument("--serial-port", dest="serial_port", default="COM3",
                   help="Serial port for MODBUS_RTU")
    p.add_argument("--prim_path", type=str, default=None,
                   help='Isaac Sim gripper prim path, e.g. "/World/piab_picobot"')
    args = p.parse_args()

    main(ip=args.ip,
         serial_port=args.serial_port,
         protocol=args.protocol,
         prim_path=args.prim_path)
