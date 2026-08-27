"""
Demonstrates opening a parallel gripper.

Supports OnRobot and Robotiq grippers, and Isaac Sim.

Usage:
    python open.py --ip <GRIPPER_IP>
    python open.py --prim_path <PRIM_PATH>

Note:
    Experimental gripper support for Isaac Sim:  Robotiq 2F85 (USD-based simulation only); Schunk EGP and
    PZN+ are.
"""

import argparse
from loguru import logger

from telekinesis.synapse.tools.parallel_grippers import onrobot


def main(ip: str | None,
         protocol: str,
         prim_path: str | None) -> None:
    """Opens an OnRobot gripper fully at 50 N force."""

    #===================== Create Gripper ======================================
    gripper = onrobot.OnRobotRG6()

    try:
        #===================== Connect Gripper =================================
        if prim_path:
            gripper.connect(simulation_prim_path=prim_path)
        else:
            gripper.connect(ip=ip, protocol=protocol)

        # ==================== Run Skill ====================================
        status = gripper.open(force=50.0,
                              asynchronous=False)
        logger.success(f"open() status: {status}, "
                       f"position: {gripper.get_current_position():.2f}")
    except (ConnectionError, OSError) as e:
        logger.error(f"Error occurred: {e}")
    finally:
        gripper.disconnect()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="OnRobot gripper open")
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
