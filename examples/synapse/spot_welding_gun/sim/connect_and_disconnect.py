"""Connect to and disconnect from a spot-welding gun in Isaac Sim.

Usage:
    python connect_and_disconnect.py
    python connect_and_disconnect.py --simulation-prim-path /World/spot_welding_gun_modelled
"""

import argparse

from telekinesis.synapse.tools.welding_gun import SpotWeldingGun


def main(simulation_prim_path: str) -> None:
    """Connect to the simulated gun and release the connection.

    Args:
        simulation_prim_path: USD path of the welding-gun articulation.
    """
    gun = SpotWeldingGun()
    try:
        gun.connect(simulation_prim_path=simulation_prim_path)
        print(f"Connected: {gun.is_connected}")
    finally:
        if gun.is_connected:
            gun.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Connect to a spot-welding gun simulated in Isaac Sim.",
    )
    parser.add_argument(
        "--simulation-prim-path",
        default="/World/spot_welding_gun_modelled",
        help="USD path of the simulated spot-welding gun.",
    )
    arguments = parser.parse_args()
    main(simulation_prim_path=arguments.simulation_prim_path)
