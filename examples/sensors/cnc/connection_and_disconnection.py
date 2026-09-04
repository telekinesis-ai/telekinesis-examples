"""
Demonstrates connecting to and disconnecting from a CNC machine's door in a
running Isaac Sim stage.

Supports Isaac Sim only.

Usage:
    python connection_and_disconnection.py --prim_path <PRIM_PATH>

Note:
    Open Isaac Sim before running this. A stage that does not hold the
    door prim gets the bundled demo CNC machine added to it at
    /World/cnc_machine, keeping whatever is already in the stage; a machine
    of your own needs its door as its own prim, named with --prim_path.

    Both positions are relative to the door prim's parent: select the door
    prim, slide it fully open and fully closed in the stage, and read the
    position off Property > Transform each time. Where that transform is
    shown as a matrix, the first three values of the bottom row are its x,
    y and z.
"""

import argparse

from loguru import logger

from telekinesis.medulla.machines import isaacsim


def main(prim_path: str,
         open_position: list[float],
         closed_position: list[float]) -> None:
    """Connects to a CNC machine's door, then disconnects."""

    # ===================== Create Machine =======================================
    machine = isaacsim.CNCMachine(name="my_simulated_cnc_machine")
    machine.set_usd("https://assets.telekinesis.ai/usd/machines/cnc_machine.zip")

    try:
        # ==================== Run Skill ============================================
        machine.connect(simulation_prim_path=prim_path,
                        open_position=open_position,
                        closed_position=closed_position)
        logger.success(f"Connected: {machine.is_connected}.")
    except (ConnectionError, RuntimeError) as e:
        logger.error(f"Error occurred: {e}")
    finally:
        machine.disconnect()


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Connect to and disconnect from a CNC machine's door in Isaac Sim")
    p.add_argument("--prim_path", type=str, default="/World/cnc_machine/E_body_1/door",
                   help='Isaac Sim CNC machine door prim path, e.g. '
                        '"/World/cnc_machine/E_body_1/door"')
    p.add_argument("--open_position", type=float, nargs=3,
                   default=[-0.68654, -0.05313, 1.208], metavar=("X", "Y", "Z"),
                   help="Door position in meters, relative to the door "
                        "prim's parent, at which it stands open")
    p.add_argument("--closed_position", type=float, nargs=3,
                   default=[-0.2193, -0.05313, 1.208], metavar=("X", "Y", "Z"),
                   help="Door position in meters, relative to the door "
                        "prim's parent, at which it stands closed")
    args = p.parse_args()

    main(prim_path=args.prim_path,
         open_position=args.open_position,
         closed_position=args.closed_position)
