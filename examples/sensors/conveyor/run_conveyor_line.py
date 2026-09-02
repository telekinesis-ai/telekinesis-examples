"""
Runs a line of conveyor belts in a running Isaac Sim stage for a few seconds.

Supports Isaac Sim only. Each belt runs at the speed the scene authored
unless ``start()`` is given one, and a negative speed runs it backwards.

Usage:
    python run_conveyor_line.py
    python run_conveyor_line.py --prim_paths /World/ConveyorBelt_A08 /World/ConveyorBelt_A11

Note:
    Open Isaac Sim and add conveyor prims before running this. If there are
    none in the stage yet, follow
    https://docs.isaacsim.omniverse.nvidia.com/6.0.1/digital_twin/warehouse_logistics/ext_isaacsim_asset_gen_conveyor.html
    to create one. If a belt connects but never moves, check that its
    ``ConveyorNode`` has ``inputs:enabled`` set to true. Add an object on
    top of the line beforehand to see it carried along.
"""

import argparse
import time

from loguru import logger

from telekinesis.medulla.conveyors import isaacsim


def connect_to_conveyors(prim_paths: list[str], cargo_root: str | None):
    conveyor_list = []
    for path in prim_paths:
        belt = isaacsim.IsaacSimConveyor(name=path)
        belt.connect(simulation_prim_path=path, cargo_root=cargo_root)
        conveyor_list.append(belt)

    return conveyor_list


def start_conveyors(conveyor_list, velocity: float, run_seconds: float) -> None:
    for belt in conveyor_list:
        belt.start(velocity=velocity)
        logger.info(
            f"Running {belt.name} at {belt.velocity} m/s for {run_seconds} s."
        )


def stop_conveyors(conveyor_list) -> None:
    for belt in conveyor_list:
        belt.stop()
        logger.success(f"Stopped {belt.name}. Running: {belt.is_running}.")


def disconnect_conveyors(conveyor_list) -> None:
    for belt in conveyor_list:
        belt.disconnect()
        logger.info(f"Disconnected {belt.name}.")


def main(prim_paths: list[str],
         cargo_root: str | None,
         velocity: float,
         run_seconds: float) -> None:
    """Runs a line of conveyor belts for a few seconds, then stops them."""

    #===================== Create and Connect Conveyors =========================
    conveyors = []
    try:
        conveyors = connect_to_conveyors(prim_paths, cargo_root)

        # ==================== Run Skill ============================================
        start_conveyors(conveyors, velocity, run_seconds)
        time.sleep(run_seconds)

        stop_conveyors(conveyors)
    except (ConnectionError, RuntimeError) as e:
        logger.error(f"Error occurred: {e}")
    finally:
        disconnect_conveyors(conveyors)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Run a conveyor line in Isaac Sim")
    p.add_argument("--prim_paths", type=str, nargs="+",
                   default=[
                       "/World/ConveyorBelt_A08",
                       "/World/ConveyorBelt_A11",
                       "/World/ConveyorBelt_A08_01",
                   ],
                   help='Isaac Sim conveyor prim paths to run, e.g. '
                        '"/World/ConveyorBelt_A08"')
    p.add_argument("--cargo_root", type=str, default="/World",
                   help="USD path whose rigid bodies are woken when the belts start")
    p.add_argument("--velocity", type=float, default=0.5,
                   help="Signed speed in meters per second to run the belts at")
    p.add_argument("--run_seconds", type=float, default=4.3,
                   help="How long to run the belts before stopping")
    args = p.parse_args()

    main(prim_paths=args.prim_paths,
         cargo_root=args.cargo_root,
         velocity=args.velocity,
         run_seconds=args.run_seconds)
