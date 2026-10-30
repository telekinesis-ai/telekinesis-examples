"""
Demonstrates tuning how a suction gripper's grip behaves.

Supports Isaac Sim only.

Usage:
    python set_properties.py --prim_path <PRIM_PATH>

Note:
    An argument left out keeps its current value. The force limits, grip
    distance and retry interval apply immediately; forward_axis and the
    limits apply the next time the timeline is stopped and replayed.
"""

import argparse
from loguru import logger

from telekinesis.synapse.tools.suction_grippers import isaacsim


def main(prim_path: str) -> None:
    """Raises the force limits of a gripper and lengthens its reach."""

    #===================== Create Gripper ======================================
    gripper = isaacsim.SuctionGripper()

    try:
        #===================== Connect Gripper =================================
        gripper.connect(simulation_prim_path=prim_path)

        # ==================== Run Skill ====================================
        properties = gripper.set_properties(coaxial_force_limit=200.0,
                                            shear_force_limit=100.0,
                                            max_grip_distance=0.02)
        logger.success(f"Grip properties after the write: {properties}")
    except (ConnectionError, OSError) as e:
        logger.error(f"Error occurred: {e}")
    finally:
        if gripper.is_connected:
            gripper.disconnect()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Suction gripper set properties")
    p.add_argument("--prim_path", type=str, required=True,
                   help='Isaac Sim gripper prim path, e.g. "/World/suction_gripper"')
    args = p.parse_args()

    main(prim_path=args.prim_path)
