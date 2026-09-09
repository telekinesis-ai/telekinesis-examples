"""
Demonstrates loading a suction gripper into the stage from a chosen USD asset.

Supports Isaac Sim only, so the example needs an Isaac Sim gripper prim path;
without one it reports that and exits, which is why running it as part of a
hardware run of run_all_examples.py does nothing.

Note:
    Connecting to a prim path that is not in the open stage loads the gripper
    there from the asset set here, so it does not have to be imported by hand
    first. A prim path that is already in the stage is used as it is and the
    asset is ignored, so this runs against either stage.

Usage:
    python set_usd.py --prim_path <PRIM_PATH>
    python set_usd.py --prim_path <PRIM_PATH> --usd <BUNDLE_URL_OR_PATH>
"""

import argparse
from loguru import logger

from telekinesis.synapse.tools.suction_grippers import isaacsim


def main(prim_path: str | None, usd: str) -> None:
    """Loads the gripper from the given USD asset and reports where it came from."""

    if not prim_path:
        logger.warning(
            "set_usd() loads a gripper into an Isaac Sim stage, so this "
            "example needs --prim_path. Skipping."
        )
        return

    #===================== Create Gripper ======================================
    gripper = isaacsim.SuctionGripper()

    try:
        # ==================== Run Skill =======================================
        gripper.set_usd(usd)

        #===================== Connect Gripper =================================
        gripper.connect(simulation_prim_path=prim_path)

        logger.success(f"Gripper at {prim_path} is loaded from {gripper.usd_path}.")
    except (ConnectionError, OSError) as e:
        logger.error(f"Error occurred: {e}")
    finally:
        if gripper.is_connected:
            gripper.disconnect()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Suction gripper set USD asset")
    p.add_argument("--prim_path", type=str, default=None,
                   help='Isaac Sim gripper prim path, e.g. "/World/suction_gripper"')
    p.add_argument("--usd", type=str,
                   default="tools/suction_grippers/defitech_modelled_surface_gripper",
                   help="Bundle on the asset server, an HTTP(S) URL of a .zip "
                        "bundle, or a path on this machine to a .usd file, a "
                        "bundle directory or a .zip")
    args = p.parse_args()

    main(prim_path=args.prim_path,
         usd=args.usd)
