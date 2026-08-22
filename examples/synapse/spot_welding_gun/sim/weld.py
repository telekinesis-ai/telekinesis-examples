"""Perform one simulated spot-welding cycle in Isaac Sim.

Usage:
    python weld.py
    python weld.py --duration-seconds 1.0
"""

import argparse

from loguru import logger

from telekinesis.synapse.tools.welding_gun import SpotWeldingGun


SPARK_PRIM_RELATIVE_PATHS = (
    "base_link/base_visual/mountplate/SpotWeldingTool_U20__U23_3/spark1",
    "base_link/base_visual/mountplate/SpotWeldingTool_U20__U23_3/spark2",
)


def main(simulation_prim_path: str, duration_seconds: float) -> None:
    """Connect to the simulated gun and perform one weld cycle.

    Args:
        simulation_prim_path: USD path of the welding-gun articulation.
        duration_seconds: Number of seconds to display the weld sparks.
    """
    gun_root = simulation_prim_path.rstrip("/")
    gun = SpotWeldingGun(
        weld_duration_seconds=duration_seconds,
        spark_prim_paths=tuple(
            f"{gun_root}/{relative_path}" for relative_path in SPARK_PRIM_RELATIVE_PATHS
        ),
    )
    try:
        gun.connect(simulation_prim_path=simulation_prim_path)
        gun.weld()
        logger.success("Weld cycle completed: closed, sparked, and reopened.")
    finally:
        if gun.is_connected:
            gun.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Perform one spot-welding cycle in Isaac Sim.",
    )
    parser.add_argument(
        "--simulation-prim-path",
        default="/World/spot_welding_gun_modelled",
        help="USD path of the simulated spot-welding gun.",
    )
    parser.add_argument(
        "--duration-seconds",
        type=float,
        default=0.5,
        help="Number of seconds to display both spark prims.",
    )
    arguments = parser.parse_args()
    main(
        simulation_prim_path=arguments.simulation_prim_path,
        duration_seconds=arguments.duration_seconds,
    )
