"""Open a spot-welding gun simulated in Isaac Sim.

Usage:
    python open.py
    python open.py --simulation-prim-path /World/spot_welding_gun_modelled
"""

import argparse

from loguru import logger

from telekinesis.synapse.tools.welding_gun import SpotWeldingGun


def main(simulation_prim_path: str) -> None:
    """Connect to the simulated gun and open its electrode.

    Args:
        simulation_prim_path: USD path of the welding-gun articulation.
    """
    gun = SpotWeldingGun()
    try:
        gun.connect(simulation_prim_path=simulation_prim_path)
        status = gun.open()
        logger.success(f"open() status: {status}")
    finally:
        if gun.is_connected:
            gun.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Open a spot-welding gun simulated in Isaac Sim.",
    )
    parser.add_argument(
        "--simulation-prim-path",
        default="/World/spot_welding_gun_modelled",
        help="USD path of the simulated spot-welding gun.",
    )
    arguments = parser.parse_args()
    main(simulation_prim_path=arguments.simulation_prim_path)
