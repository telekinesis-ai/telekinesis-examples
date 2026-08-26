"""
Demonstrates setting the stroke range of a parallel gripper.

Supports OnRobot and Robotiq grippers, and Isaac Sim.

Usage:
    python set_position_range.py --ip <GRIPPER_IP>
    python set_position_range.py --prim_path <PRIM_PATH>
"""

import argparse
from loguru import logger

from telekinesis.synapse.tools.parallel_grippers import onrobot


def main(ip: str | None,
         protocol: str,
         prim_path: str | None) -> None:
    """Sets the stroke range of an OnRobot gripper to 85 mm."""

    #===================== Create Gripper ======================================
    gripper = onrobot.OnRobotRG6()

    try:
        #===================== Connect Gripper =================================
        if prim_path:
            gripper.connect(simulation_prim_path=prim_path)
        else:
            gripper.connect(ip=ip, protocol=protocol)

        # ==================== Run Skill ====================================
        gripper.set_position_range_mm(position_range_mm=85.0)
        logger.success("Position range set to 85 mm.")
    except (ConnectionError, OSError) as e:
        logger.error(f"Error occurred: {e}")
    finally:
        gripper.disconnect()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="OnRobot gripper set position range")
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
