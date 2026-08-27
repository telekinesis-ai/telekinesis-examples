"""
Demonstrates grasping an object with a suction gripper.

Supports Piab grippers.

Usage:
    python grasp.py --ip <ROBOT_IP>
    python grasp.py --protocol MODBUS_RTU --serial-port COM3
    python grasp.py --prim_path <PRIM_PATH>

"""

import argparse
import time
from loguru import logger

from telekinesis.synapse.tools.suction_grippers import piab


def main(ip: str | None,
         serial_port: str,
         protocol: str,
         prim_path: str | None) -> None:
    """Grasps an object with a Piab gripper at a 60% vacuum level."""

    #===================== Create Gripper ======================================
    gripper = piab.PiabPiCobotElectric()

    # ==================== Run Skill ===========================================
    try:
        #===================== Connect Gripper =================================
        if prim_path:
            gripper.connect(simulation_prim_path=prim_path)
        else:
            gripper.connect(ip=ip, serial_port=serial_port, protocol=protocol)

        gripper.grasp(vacuum_level=60, unit="percentage")
        logger.success("Grasp command issued.")

        # Give the pump time to build up vacuum before reading the status.
        time.sleep(2.0)
        logger.success(f"Part present: {gripper.get_part_present()}")
    except (ConnectionError, OSError) as e:
        logger.error(f"Error occurred: {e}")
    finally:
        gripper.disconnect()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Piab gripper grasp")
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
