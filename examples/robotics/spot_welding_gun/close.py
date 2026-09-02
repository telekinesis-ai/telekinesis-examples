"""
Demonstrates closing a spot-welding gun.

Supports Isaac Sim only.

Usage:
    python close.py --prim_path <PRIM_PATH>
"""

import argparse
from loguru import logger

from telekinesis.synapse.tools import welding_gun


def main(prim_path: str) -> None:
    """Closes a spot-welding gun electrode."""

    #===================== Create Gripper ======================================
    gun = welding_gun.SpotWeldingGun()

    try:
        #===================== Connect Gripper =================================
        gun.connect(simulation_prim_path=prim_path)

        # ==================== Run Skill ====================================
        status = gun.close()
        logger.success(f"close() status: {status}")
    except (ConnectionError, OSError) as e:
        logger.error(f"Error occurred: {e}")
    finally:
        if gun.is_connected:
            gun.disconnect()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Spot-welding gun close")
    p.add_argument("--prim_path", type=str, default="/World/spot_welding_gun_modelled",
                   help='Isaac Sim gun prim path, e.g. "/World/spot_welding_gun_modelled"')
    args = p.parse_args()

    main(prim_path=args.prim_path)
