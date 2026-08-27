"""
Demonstrates reading whether a suction gripper currently holds an object.

Supports Piab grippers on MODBUS_RTU protocol.

Usage:
    python get_part_present.py --serial-port COM3
    python get_part_present.py --prim_path <PRIM_PATH>
"""

import argparse
from loguru import logger

from telekinesis.synapse.tools.suction_grippers import piab


def main(serial_port: str, prim_path: str | None) -> None:
    """Reads whether a Piab gripper currently holds an object."""

    #===================== Create Gripper ======================================
    gripper = piab.PiabPiCobotElectric()

    # ==================== Run Skill ===========================================
    try:
        #===================== Connect Gripper =================================
        if prim_path:
            gripper.connect(simulation_prim_path=prim_path)
        else:
            gripper.connect(serial_port=serial_port, protocol="MODBUS_RTU")

        logger.success(f"Part present: {gripper.get_part_present()}")
    except (ConnectionError, OSError) as e:
        logger.error(f"Error occurred: {e}")
    finally:
        gripper.disconnect()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Piab gripper get part present")
    p.add_argument("--serial-port", dest="serial_port", default="COM3",
                   help="Serial port for MODBUS_RTU")
    p.add_argument("--prim_path", type=str, default=None,
                   help='Isaac Sim gripper prim path, e.g. "/World/piab_picobot"')
    args = p.parse_args()

    main(serial_port=args.serial_port, prim_path=args.prim_path)
