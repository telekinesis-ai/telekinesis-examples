"""
Demonstrates connecting to and disconnecting from a conveyor belt in a
running Isaac Sim stage.

Supports Isaac Sim only.

Usage:
    python connection_and_disconnection.py --prim_path <PRIM_PATH>
    python connection_and_disconnection.py --prim_path <PRIM_PATH> --load_usd

Note:
    Open Isaac Sim and add a conveyor prim before running this. If there is
    none in the stage yet, follow
    https://docs.isaacsim.omniverse.nvidia.com/6.0.1/digital_twin/warehouse_logistics/ext_isaacsim_asset_gen_conveyor.html
    to create one, or pass --load_usd to add one to the open stage -- this
    keeps whatever is already in the stage.
"""

import argparse

from loguru import logger

from telekinesis.medulla.conveyors import isaacsim


def main(prim_path: str, cargo_root: str | None, load_usd: bool) -> None:
    """Connects to a conveyor belt, then disconnects."""

    if load_usd:
        # ===================== Load Demo Scene (Optional) ===========================
        from telekinesis import datatypes, isaacsim_client

        client = isaacsim_client.IsaacSimClient(
            api_key="",
            base_url="http://127.0.0.1:8766",
            websocket_base_url="ws://127.0.0.1:8766",
        )
        asset = datatypes.USD.from_url(
            "https://assets.telekinesis.ai/usd/conveyors/simple_conveyor.zip"
        )
        client.stage.add_to_scene(uri=asset.path.as_posix(),
                                  prim_path="/World/simple_conveyor")

    # ===================== Create Conveyor ======================================
    belt = isaacsim.Conveyor(name="my_simulated_conveyor")

    try:
        # ==================== Run Skill ============================================
        belt.connect(simulation_prim_path=prim_path, cargo_root=cargo_root)
        logger.success(f"Connected: {belt.is_connected}.")
    except (ConnectionError, RuntimeError) as e:
        logger.error(f"Error occurred: {e}")
    finally:
        belt.disconnect()


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Connect to and disconnect from a conveyor belt in Isaac Sim")
    p.add_argument("--prim_path", type=str, default="/World/simple_conveyor",
                   help='Isaac Sim conveyor prim path, e.g. "/World/simple_conveyor"')
    p.add_argument("--cargo_root", type=str, default="/World",
                   help="USD path whose rigid bodies are woken when the belt starts")
    p.add_argument("--load_usd", action=argparse.BooleanOptionalAction, default=False,
                   help="Add the bundled demo conveyor to the open stage at "
                        "/World/simple_conveyor before connecting. Use this "
                        "if you don't already have one in the stage.")
    args = p.parse_args()

    main(prim_path=args.prim_path, cargo_root=args.cargo_root, load_usd=args.load_usd)
