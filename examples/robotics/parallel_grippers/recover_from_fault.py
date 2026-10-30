"""
Demonstrates clearing a fault on a parallel gripper and reactivating it.

Supports only Robotiq grippers (Modbus RTU), and Isaac Sim.

Usage:
    python recover_from_fault.py --protocol MODBUS_RTU --serial-port COM4
    python recover_from_fault.py --prim_path <PRIM_PATH>

Note:
    The recovery sweeps the fingers fully open and closed, releasing any held
    object. Run it with nothing between the fingers. A simulated gripper has
    no faults, so recover_from_fault() is a no-op when connected to Isaac Sim.
"""

import argparse
from loguru import logger

from telekinesis.synapse.tools.parallel_grippers import robotiq


def main(serial_port: str,
         protocol: str,
         prim_path: str | None) -> None:
    """Resets and reactivates a Robotiq gripper."""

    #===================== Create Gripper ======================================
    gripper = robotiq.Robotiq2F85()

    try:
        #===================== Connect Gripper =================================
        if prim_path:
            gripper.connect(simulation_prim_path=prim_path)
        else:
            gripper.connect(serial_port=serial_port, protocol=protocol)

        # ==================== Run Skill ====================================
        gripper.recover_from_fault()
        logger.success("Gripper reset and reactivated.")
    except (ConnectionError, OSError) as e:
        logger.error(f"Error occurred: {e}")
    finally:
        gripper.disconnect()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Robotiq gripper recover from fault")
    p.add_argument("--protocol", choices=["MODBUS_RTU"], default="MODBUS_RTU")
    p.add_argument("--serial-port", dest="serial_port", default="COM4",
                   help="Serial port for MODBUS_RTU")
    p.add_argument("--prim_path", type=str, default=None,
                   help='Isaac Sim gripper prim path, e.g. "/World/robotiq_2f85"')
    args = p.parse_args()

    main(serial_port=args.serial_port,
         protocol=args.protocol,
         prim_path=args.prim_path)
