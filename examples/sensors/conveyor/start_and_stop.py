"""
Demonstrates starting and stopping a conveyor belt in a running Isaac Sim
stage.

Supports Isaac Sim only. The belt runs at the speed the scene authored
unless ``start()`` is given one, and a negative speed runs it backwards. It
keeps that speed once stopped, so starting it again without a velocity
resumes at the same speed.

Usage:
    python start_and_stop.py --prim_path <PRIM_PATH>
    python start_and_stop.py --prim_path <PRIM_PATH> --velocity -0.5

Note:
    Open Isaac Sim and add a conveyor prim before running this. If there is
    none in the stage yet, follow
    https://docs.isaacsim.omniverse.nvidia.com/6.0.1/digital_twin/warehouse_logistics/ext_isaacsim_asset_gen_conveyor.html
    to create one. If the belt connects but never moves, check that its
    ``ConveyorNode`` has ``inputs:enabled`` set to true. Add an object on
    top of the belt beforehand to see it carried along.
"""

import argparse

from loguru import logger

from telekinesis.medulla.conveyors import isaacsim


def main(prim_path: str, cargo_root: str | None, velocity: float | None) -> None:
    """Starts a conveyor belt, then stops it."""

    # ===================== Create Conveyor ======================================
    belt = isaacsim.Conveyor(name="my_simulated_conveyor")

    try:
        # ===================== Connect Conveyor ==================================
        belt.connect(simulation_prim_path=prim_path, cargo_root=cargo_root)

        # ==================== Run Skill ============================================
        belt.start(velocity=velocity)
        logger.success(f"Running at {belt.velocity} m/s. Running: {belt.is_running}.")

        belt.stop()
        logger.success(f"Stopped. Running: {belt.is_running}.")
    except (ConnectionError, RuntimeError) as e:
        logger.error(f"Error occurred: {e}")
    finally:
        belt.disconnect()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Start and stop a conveyor belt in Isaac Sim")
    p.add_argument("--prim_path", type=str, default="/World/ConveyorTrack",
                   help='Isaac Sim conveyor prim path, e.g. "/World/ConveyorTrack"')
    p.add_argument("--cargo_root", type=str, default="/World",
                   help="USD path whose rigid bodies are woken when the belt starts")
    p.add_argument("--velocity", type=float, default=None,
                   help="Signed speed in meters per second to run the belt at. "
                        "Defaults to the belt's own configured speed")
    args = p.parse_args()

    main(prim_path=args.prim_path, cargo_root=args.cargo_root, velocity=args.velocity)
