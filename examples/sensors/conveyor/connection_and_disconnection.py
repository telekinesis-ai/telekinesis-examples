"""
Demonstrates connecting to and disconnecting from a conveyor belt in a
running Isaac Sim stage.

Supports Isaac Sim only.

Usage:
    python connection_and_disconnection.py --prim_path <PRIM_PATH>

Note:
    Open Isaac Sim and add a conveyor prim before running this. If there is
    none in the stage yet, follow
    https://docs.isaacsim.omniverse.nvidia.com/6.0.1/digital_twin/warehouse_logistics/ext_isaacsim_asset_gen_conveyor.html
    to create one.
"""

import argparse

from loguru import logger

from telekinesis.medulla.conveyors import isaacsim


def main(prim_path: str, cargo_root: str | None) -> None:
    """Connects to a conveyor belt, then disconnects."""

    # ===================== Create Conveyor ======================================
    belt = isaacsim.Conveyor(name="my_simulated_conveyor")

    try:
        # ==================== Run Skill ============================================
        belt.connect(simulation_prim_path=prim_path, cargo_root=cargo_root)
    except (ConnectionError, RuntimeError) as e:
        logger.error(f"Error occurred: {e}")
    finally:
        belt.disconnect()


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Connect to and disconnect from a conveyor belt in Isaac Sim")
    p.add_argument("--prim_path", type=str, default="/World/ConveyorTrack",
                   help='Isaac Sim conveyor prim path, e.g. "/World/ConveyorTrack"')
    p.add_argument("--cargo_root", type=str, default="/World",
                   help="USD path whose rigid bodies are woken when the belt starts")
    args = p.parse_args()

    main(prim_path=args.prim_path, cargo_root=args.cargo_root)
