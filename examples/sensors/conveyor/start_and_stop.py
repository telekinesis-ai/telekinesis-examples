"""
Runs a conveyor belt in a running Isaac Sim stage for a few seconds.

Supports Isaac Sim only. The belt runs at the speed the scene authored
unless ``start()`` is given one, and a negative speed runs it backwards.

Usage:
    python start_and_stop.py --prim_path <PRIM_PATH>
    python start_and_stop.py --prim_path <PRIM_PATH> --cargo_root <CARGO_ROOT> --velocity 0.5

Note:
    Open Isaac Sim before running this. A stage that does not hold the
    conveyor prim gets the bundled demo belt added to it at
    /World/simple_conveyor, keeping whatever is already in the stage; to
    build a belt of your own instead, follow
    https://docs.isaacsim.omniverse.nvidia.com/6.0.1/digital_twin/warehouse_logistics/ext_isaacsim_asset_gen_conveyor.html
    and name it with --prim_path.

    If the belt connects but never moves, check that its ``ConveyorNode``
    (in ConveyorBeltGraph i.e. OmniGraphNode) has ``inputs:enabled`` set to
    true. Add an object on top of the belt beforehand to see it carried
    along.
"""

import argparse
import time

from loguru import logger

from telekinesis.medulla.conveyors import isaacsim


def main(prim_path: str,
         cargo_root: str | None,
         velocity: float,
         run_seconds: float) -> None:
    """Runs a conveyor belt for a few seconds, then stops it."""

    # ===================== Create Conveyor ======================================
    belt = isaacsim.Conveyor(name="simulated_conveyor")
    belt.set_usd("https://assets.telekinesis.ai/usd/conveyors/simple_conveyor.zip")

    try:
        # ===================== Connect Conveyor ==================================
        belt.connect(simulation_prim_path=prim_path, cargo_root=cargo_root)

        # ==================== Run Skill ============================================
        belt.start(velocity=velocity)
        logger.info(f"Running at {belt.velocity} m/s for {run_seconds} s.")
        time.sleep(run_seconds)

        belt.stop()
        logger.success(f"Stopped. Running: {belt.is_running}.")
    except (ConnectionError, RuntimeError) as e:
        logger.error(f"Error occurred: {e}")
    finally:
        belt.disconnect()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Start and stop a conveyor belt in Isaac Sim")
    p.add_argument("--prim_path", type=str, default="/World/simple_conveyor",
                   help='Isaac Sim conveyor prim path, e.g. "/World/simple_conveyor"')
    p.add_argument("--cargo_root", type=str, default="/World",
                   help="USD path whose rigid bodies are woken when the belt starts")
    p.add_argument("--velocity", type=float, default=0.5,
                   help="Signed speed in meters per second to run the belt at")
    p.add_argument("--run_seconds", type=float, default=3.0,
                   help="How long to run the belt before stopping")
    args = p.parse_args()

    main(prim_path=args.prim_path,
         cargo_root=args.cargo_root,
         velocity=args.velocity,
         run_seconds=args.run_seconds)
