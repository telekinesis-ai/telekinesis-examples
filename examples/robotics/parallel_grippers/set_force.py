"""
Demonstrates setting the default grip force of a parallel gripper.

Supports OnRobot and Robotiq grippers, and Isaac Sim.

Usage:
    python set_force.py --ip <GRIPPER_IP>
    python set_force.py --prim_path <PRIM_PATH>

Note:
    The simulation does not model grip force, so in Isaac Sim this call is
    accepted but has no effect on the motion.

    Experimental gripper support for Isaac Sim:  Robotiq 2F85 (USD-based simulation only); Schunk EGP and
    PZN+ are.
"""

import argparse
from loguru import logger

from telekinesis.synapse.tools.parallel_grippers import onrobot


def main(ip: str | None,
         protocol: str,
         prim_path: str | None) -> None:
    """Sets the default grip force of an OnRobot gripper to 50 N."""

    #===================== Create Gripper ======================================
    gripper = onrobot.OnRobotRG6()

    try:
        #===================== Connect Gripper =================================
        if prim_path:
            gripper.connect(simulation_prim_path=prim_path)
        else:
            gripper.connect(ip=ip, protocol=protocol)

        # ==================== Run Skill ====================================
        actual = gripper.set_force(force=50.0)
        logger.success(f"Default force set; effective: {actual}")
    except (ConnectionError, OSError) as e:
        logger.error(f"Error occurred: {e}")
    finally:
        gripper.disconnect()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="OnRobot gripper set force")
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
