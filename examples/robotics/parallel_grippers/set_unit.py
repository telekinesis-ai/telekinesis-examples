"""
Demonstrates setting the position unit of a parallel gripper.

Supports OnRobot and Robotiq grippers, and Isaac Sim.

Usage:
    python set_unit.py --ip <GRIPPER_IP>
    python set_unit.py --protocol MODBUS_RTU --serial-port COM4
    python set_unit.py --prim_path <PRIM_PATH>

Note:
    The simulation models neither speed nor force, so in Isaac Sim only the
    position unit takes effect; the other two are validated and then ignored.
"""

import argparse
from loguru import logger

from telekinesis.synapse.tools.parallel_grippers import robotiq


def main(ip: str | None,
         serial_port: str,
         protocol: str,
         prim_path: str | None) -> None:
    """Switches the position unit of a Robotiq gripper to normalized (0..1)."""

    #===================== Create Gripper ======================================
    gripper = robotiq.Robotiq2F85()

    try:
        #===================== Connect Gripper =================================
        if prim_path:
            gripper.connect(simulation_prim_path=prim_path)
        else:
            gripper.connect(ip=ip, serial_port=serial_port, protocol=protocol)

        # ==================== Run Skill ====================================
        gripper.set_unit(parameter="position", unit="normalized")
        logger.success("Position unit set to 'normalized'.")
    except (ConnectionError, OSError) as e:
        logger.error(f"Error occurred: {e}")
    finally:
        gripper.disconnect()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Robotiq gripper set unit")
    p.add_argument("--protocol",
                   choices=["URCAP", "MODBUS_RTU"],
                   default="URCAP")
    p.add_argument("--ip", default=None, help="IP for Robotiq Gripper")
    p.add_argument("--serial-port", dest="serial_port", default="COM4",
                   help="Serial port for MODBUS_RTU")
    p.add_argument("--prim_path", type=str, default=None,
                   help='Isaac Sim gripper prim path, e.g. "/World/robotiq_2f85"')
    args = p.parse_args()

    main(ip=args.ip,
         serial_port=args.serial_port,
         protocol=args.protocol,
         prim_path=args.prim_path)
