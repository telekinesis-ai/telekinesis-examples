"""Move a simulated spot-welding electrode to a normalized position.

Usage:
    python move.py
    python move.py --position 0.5
"""

import argparse

from loguru import logger

from telekinesis.synapse.tools.welding_gun import SpotWeldingGun


def main(simulation_prim_path: str, position: float) -> None:
    """Connect to the simulated gun and move its electrode.

    Args:
        simulation_prim_path: USD path of the welding-gun articulation.
        position: Normalized joint position from ``0.0`` to ``1.0``.
    """
    gun = SpotWeldingGun()
    try:
        gun.connect(simulation_prim_path=simulation_prim_path)
        status = gun.move(position=position)
        logger.success(f"move({position}) status: {status}")
    finally:
        if gun.is_connected:
            gun.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Move a spot-welding gun simulated in Isaac Sim.",
    )
    parser.add_argument(
        "--simulation-prim-path",
        default="/World/spot_welding_gun_modelled",
        help="USD path of the simulated spot-welding gun.",
    )
    parser.add_argument(
        "--position",
        type=float,
        default=0.5,
        help="Normalized electrode position from 0.0 (open) to 1.0 (closed).",
    )
    arguments = parser.parse_args()
    main(
        simulation_prim_path=arguments.simulation_prim_path,
        position=arguments.position,
    )
