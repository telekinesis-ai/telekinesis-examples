"""
Demonstrates setting the position unit of a parallel gripper.

Supports OnRobot and Robotiq grippers, and Isaac Sim.

Usage:
    python set_unit.py --ip <GRIPPER_IP>
    python set_unit.py --prim_path <PRIM_PATH>

Note:
    OnRobot has no multi-unit backend: only parameter="position" with
    unit="mm", and parameter="force" with unit="N", are accepted. set_unit()
    is provided for signature parity with the Robotiq wrapper.

    Experimental gripper support for Isaac Sim:  Robotiq 2F85 (USD-based simulation only); Schunk EGP and
    PZN+ are.
"""

import argparse
from loguru import logger

from telekinesis.synapse.tools.parallel_grippers import onrobot


def main(ip: str | None,
         protocol: str,
         prim_path: str | None) -> None:
    """Validates the position unit of an OnRobot gripper as millimeters."""

    #===================== Create Gripper ======================================
    gripper = onrobot.OnRobotRG6()

    try:
        #===================== Connect Gripper =================================
        if prim_path:
            gripper.connect(simulation_prim_path=prim_path)
        else:
            gripper.connect(ip=ip, protocol=protocol)

        # ==================== Run Skill ====================================
        gripper.set_unit(parameter="position", unit="mm")
        logger.success("Position unit confirmed as 'mm'.")
    except (ConnectionError, OSError) as e:
        logger.error(f"Error occurred: {e}")
    finally:
        gripper.disconnect()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="OnRobot gripper set unit")
    p.add_argument("--protocol",
                   choices=["MODBUS_TCP"],
                   default="MODBUS_TCP")
    p.add_argument("--ip", default=None, help="IP for OnRobot Gripper")
    p.add_argument("--prim_path", type=str, default=None,
                   help='Isaac Sim gripper prim path, e.g. "/World/onrobot_rg6"')
    args = p.parse_args()

    main(ip=args.ip,
         protocol=args.protocol,
         prim_path=args.prim_path)
