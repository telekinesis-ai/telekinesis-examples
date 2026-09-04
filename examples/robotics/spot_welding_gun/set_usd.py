"""
Demonstrates loading a spot-welding gun into the stage from a chosen USD asset.

Supports Isaac Sim only.

Note:
    Connecting to a prim path that is not in the open stage loads the gun there
    from the asset set here, so the gun does not have to be imported by hand
    first. A prim path that is already in the stage is used as it is and the
    asset is ignored, so this runs against either stage.

Usage:
    python set_usd.py --prim_path <PRIM_PATH>
    python set_usd.py --prim_path <PRIM_PATH> --usd <BUNDLE_URL_OR_PATH>
"""

import argparse
from loguru import logger

from telekinesis.synapse.tools.welding_guns import isaacsim


def main(prim_path: str, usd: str) -> None:
    """Loads the gun from the given USD asset and reports where it came from."""

    # ===================== Create Gripper ======================================
    gun = isaacsim.SpotWeldingGun()

    try:
        # ==================== Run Skill ========================================
        gun.set_usd(usd)

        # ===================== Connect Gripper =================================
        gun.connect(simulation_prim_path=prim_path)

        logger.success(f"Gun at {prim_path} is loaded from {gun.usd_path}.")
        logger.info(f"Spark prims declared by the asset: {gun.spark_prim_paths}")
    except (ConnectionError, OSError) as e:
        logger.error(f"Error occurred: {e}")
    finally:
        if gun.is_connected:
            gun.disconnect()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Spot-welding gun set USD asset")
    p.add_argument("--prim_path", type=str, default="/World/spot_welding_gun_modelled",
                   help='Isaac Sim gun prim path, e.g. "/World/spot_welding_gun_modelled"')
    p.add_argument("--usd", type=str, default="tools/welding_guns/spot_welding_gun",
                   help="Bundle on the asset server, an HTTP(S) URL of a .zip "
                        "bundle, or a path on this machine to a .usd file, a "
                        "bundle directory or a .zip")
    args = p.parse_args()

    main(prim_path=args.prim_path,
         usd=args.usd)
