"""
Demonstrates reading how a suction gripper's grip behaves.

Supports Isaac Sim only.

Usage:
    python get_properties.py --prim_path <PRIM_PATH>
"""

import argparse
from loguru import logger

from telekinesis.synapse.tools.suction_grippers import isaacsim


def main(prim_path: str) -> None:
    """Reads the grip properties of a gripper."""

    #===================== Create Gripper ======================================
    gripper = isaacsim.SuctionGripper()

    try:
        #===================== Connect Gripper =================================
        gripper.connect(simulation_prim_path=prim_path)

        # ==================== Run Skill ====================================
        result = gripper.get_properties()
        logger.success(f"Grip properties: {result}")
    except (ConnectionError, OSError) as e:
        logger.error(f"Error occurred: {e}")
    finally:
        if gripper.is_connected:
            gripper.disconnect()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Suction gripper get_properties")
    p.add_argument("--prim_path", type=str, required=True,
                   help='Isaac Sim gripper prim path, e.g. "/World/suction_gripper"')
    args = p.parse_args()

    main(prim_path=args.prim_path)
