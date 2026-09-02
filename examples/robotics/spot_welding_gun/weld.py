"""
Demonstrates performing a spot-welding cycle with a spot-welding gun.

Supports Isaac Sim only.

Usage:
    python weld.py --prim_path <PRIM_PATH>
    python weld.py --prim_path <PRIM_PATH> --duration_seconds 1.0
"""

import argparse
from loguru import logger

from telekinesis.synapse.tools import welding_gun


def main(prim_path: str, duration_seconds: float) -> None:
    """Closes the gun, sparks for duration_seconds, and reopens it."""

    #===================== Create Gripper ======================================
    gun = welding_gun.SpotWeldingGun()

    try:
        #===================== Connect Gripper =================================
        gun.connect(simulation_prim_path=prim_path)

        # ==================== Run Skill ====================================
        gun.weld(duration_seconds=duration_seconds)
        logger.success("Weld cycle completed: closed, sparked, and reopened.")
    except (ConnectionError, OSError) as e:
        logger.error(f"Error occurred: {e}")
    finally:
        if gun.is_connected:
            gun.disconnect()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Spot-welding gun weld")
    p.add_argument("--prim_path", type=str, default="/World/spot_welding_gun_modelled",
                   help='Isaac Sim gun prim path, e.g. "/World/spot_welding_gun_modelled"')
    p.add_argument("--duration_seconds", type=float, default=0.5,
                   help="Number of seconds to display the weld sparks.")
    args = p.parse_args()

    main(prim_path=args.prim_path,
         duration_seconds=args.duration_seconds)