"""
Demonstrates tuning the individual suction cups of a suction gripper.

Supports Isaac Sim only.

Usage:
    python set_attachment_point_properties.py --prim_path <PRIM_PATH>

Note:
    An argument left out keeps its current value. The drive and the limits
    apply immediately; clearance_offset and forward_axis apply the next time
    the timeline is stopped and replayed.
"""

import argparse
from loguru import logger

from telekinesis.synapse.tools.suction_grippers import isaacsim


def main(prim_path: str) -> None:
    """Stiffens the pull-in drive and widens the clearance of every suction cup."""

    #===================== Create Gripper ======================================
    gripper = isaacsim.SuctionGripper()

    try:
        #===================== Connect Gripper =================================
        gripper.connect(simulation_prim_path=prim_path)

        # ==================== Run Skill ====================================
        cups = gripper.set_attachment_point_properties(
            z_axis_translation_drive_stiffness=1000.0,
            z_axis_translation_drive_damping=100.0,
            clearance_offset=0.005,
        )
        logger.success(f"Suction cups after the write: {cups}")
    except (ConnectionError, OSError) as e:
        logger.error(f"Error occurred: {e}")
    finally:
        if gripper.is_connected:
            gripper.disconnect()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Suction gripper set attachment point properties")
    p.add_argument("--prim_path", type=str, required=True,
                   help='Isaac Sim gripper prim path, e.g. "/World/suction_gripper"')
    args = p.parse_args()

    main(prim_path=args.prim_path)
