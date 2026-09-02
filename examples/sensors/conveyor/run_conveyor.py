"""
Runs a conveyor belt in a running Isaac Sim stage for a few seconds.

Supports Isaac Sim only. The belt runs at the speed the scene authored
unless ``start()`` is given one, and a negative speed runs it backwards.

Usage:
    python run_conveyor.py --prim_path <PRIM_PATH>
    python run_conveyor.py --prim_path <PRIM_PATH> --cargo_root <CARGO_ROOT> --velocity 0.5

Note:
    Open Isaac Sim and add a conveyor prim before running this. If there is
    none in the stage yet, follow
    https://docs.isaacsim.omniverse.nvidia.com/6.0.1/digital_twin/warehouse_logistics/ext_isaacsim_asset_gen_conveyor.html
    to create one. If the belt connects but never moves, check that its
    ``ConveyorNode`` has ``inputs:enabled`` set to true. Add an object on
    top of the belt beforehand to see it carried along.
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

    #===================== Create Conveyor ======================================
    belt = isaacsim.IsaacSimConveyor(name="simulated_conveyor")

    try:
        #===================== Connect Conveyor ==================================
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
    p = argparse.ArgumentParser(description="Run a conveyor belt in Isaac Sim")
    p.add_argument("--prim_path", type=str, default="/World/ConveyorTrack",
                   help='Isaac Sim conveyor prim path, e.g. "/World/ConveyorTrack"')
    p.add_argument("--cargo_root", type=str, default="/World",
                   help="USD path whose rigid bodies are woken when the belt starts")
    p.add_argument("--velocity", type=float, default=0.5,
                   help="Signed speed in meters per second to run the belt at")
    p.add_argument("--run_seconds", type=float, default=5.0,
                   help="How long to run the belt before stopping")
    args = p.parse_args()

    main(prim_path=args.prim_path,
         cargo_root=args.cargo_root,
         velocity=args.velocity,
         run_seconds=args.run_seconds)
