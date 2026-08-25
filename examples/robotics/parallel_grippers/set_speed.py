"""
Demonstrates setting the default speed of a parallel gripper.

Supports only Robotiq grippers, and Isaac Sim.

Usage:
    python set_speed.py --ip <GRIPPER_IP>
    python set_speed.py --protocol MODBUS_RTU --serial-port COM4
    python set_speed.py --prim_path <PRIM_PATH>

Note:
    The simulation does not model gripper speed, so in Isaac Sim this call is
    accepted but has no effect on the motion.
"""

import argparse
from loguru import logger

from telekinesis.synapse.tools.parallel_grippers import robotiq


def main(ip: str | None,
         serial_port: str,
         protocol: str,
         prim_path: str | None) -> None:
    """Sets the default speed of a Robotiq gripper to 50%."""

    #===================== Create Gripper ======================================
    gripper = robotiq.Robotiq2F85()

    try:
        #===================== Connect Gripper =================================
        if prim_path:
            gripper.connect(simulation_prim_path=prim_path)
        else:
            gripper.connect(ip=ip, serial_port=serial_port, protocol=protocol)

        # ==================== Run Skill ====================================
        actual = gripper.set_speed(speed=50.0)
        logger.success(f"Default speed set; effective: {actual}")
    except (ConnectionError, OSError) as e:
        logger.error(f"Error occurred: {e}")
    finally:
        gripper.disconnect()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Robotiq gripper set speed")
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
