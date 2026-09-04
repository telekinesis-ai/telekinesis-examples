"""
Demonstrates moving a spot-welding gun electrode to a target position.

Supports Isaac Sim only.

Usage:
    python move.py --prim_path <PRIM_PATH>
    python move.py --prim_path <PRIM_PATH> --position 0.5
"""

import argparse
from loguru import logger

from telekinesis.synapse.tools.welding_guns import isaacsim


def main(prim_path: str, position: float) -> None:
    """Moves a spot-welding gun electrode to a normalized position."""

    # ===================== Create Gripper ======================================
    gun = isaacsim.SpotWeldingGun()

    try:
        # ===================== Connect Gripper =================================
        gun.connect(simulation_prim_path=prim_path)

        # ==================== Run Skill ====================================
        status = gun.move(position=position)
        logger.success(f"move({position}) status: {status}")
    except (ConnectionError, OSError) as e:
        logger.error(f"Error occurred: {e}")
    finally:
        if gun.is_connected:
            gun.disconnect()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Spot-welding gun move")
    p.add_argument("--prim_path", type=str, default="/World/spot_welding_gun_modelled",
                   help='Isaac Sim gun prim path, e.g. "/World/spot_welding_gun_modelled"')
    p.add_argument("--position", type=float, default=0.5,
                   help="Normalized electrode position from 0.0 (open) to 1.0 (closed).")
    args = p.parse_args()

    main(prim_path=args.prim_path,
         position=args.position)
