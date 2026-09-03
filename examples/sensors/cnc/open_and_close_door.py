"""
Opens and closes the door of a CNC machine in a running Isaac Sim stage.

Supports Isaac Sim only. The door slides between the two positions given
here, easing in and out of the travel, and each move blocks until the door
arrives.

Usage:
    python open_and_close_door.py --prim_path <PRIM_PATH>
    python open_and_close_door.py --prim_path <PRIM_PATH>
        --open_position 0.0 0.0 0.0 --closed_position -0.8 0.0 0.0
    python open_and_close_door.py --prim_path <PRIM_PATH> --load_usd

Note:
    Open Isaac Sim and add a CNC machine whose door is its own prim before
    running this, or pass --load_usd to add one to the open stage -- this
    keeps whatever is already in the stage. Both positions
    are relative to the door prim's parent: select the door prim, slide it
    fully open and fully closed in the stage, and read the position off
    Property > Transform each time. Where that transform is shown as a
    matrix, the first three values of the bottom row are its x, y and z.

    The door is slid by writing its transform, so keep its path clear -- it
    passes through whatever stands in the way.
"""

import argparse

from loguru import logger

from telekinesis.medulla.machines import isaacsim


def main(prim_path: str,
         open_position: list[float],
         closed_position: list[float],
         load_usd: bool) -> None:
    """Opens the door of a CNC machine, then closes it again."""

    if load_usd:
        # ===================== Load Demo Scene (Optional) ===========================
        from telekinesis import datatypes, isaacsim_client

        client = isaacsim_client.IsaacSimClient(
            api_key="",
            base_url="http://127.0.0.1:8766",
            websocket_base_url="ws://127.0.0.1:8766",
        )
        asset = datatypes.USD.from_url(
            "https://assets.telekinesis.ai/usd/machines/cnc_machine.zip"
        )
        client.stage.add_to_scene(uri=asset.path.as_posix(),
                                  prim_path="/World/cnc_machine")

    # ===================== Create Machine =======================================
    machine = isaacsim.CNCMachine(name="my_simulated_cnc_machine")

    try:
        # ===================== Connect Machine ==================================
        machine.connect(simulation_prim_path=prim_path,
                        open_position=open_position,
                        closed_position=closed_position)

        # ==================== Run Skill ============================================
        logger.info("Opening the door.")
        machine.open()
        logger.info(f"Door open at {machine.door_position} m.")

        logger.success("Closing the door.")
        machine.close()
        logger.success(f"Door closed at {machine.door_position} m.")

        logger.info("Opening the door.")
        machine.open()
        logger.info(f"Door open at {machine.door_position} m.")
    except (ConnectionError, RuntimeError) as e:
        logger.error(f"Error occurred: {e}")
    finally:
        machine.disconnect()


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Open and close a CNC machine's door in Isaac Sim")
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
    p.add_argument("--load_usd", action=argparse.BooleanOptionalAction, default=False,
                   help="Add the bundled demo CNC machine to the open stage "
                        "at /World/cnc_machine before connecting. Use this "
                        "if you don't already have one in the stage.")
    args = p.parse_args()

    main(prim_path=args.prim_path,
         open_position=args.open_position,
         closed_position=args.closed_position,
         load_usd=args.load_usd)
