"""Load a CNC machine into an Isaac Sim stage from a USD asset.

Supports Isaac Sim only.

Usage:
    python set_usd.py --prim_path <PRIM_PATH>
    python set_usd.py --prim_path <PRIM_PATH> --usd <BUNDLE_URL_OR_PATH>

Note:
    Open Isaac Sim before running this. The asset is loaded only when the door
    prim is missing from the open stage. An existing prim is used as it is and
    the configured asset is ignored.

    The open and closed positions are in meters relative to the door prim's
    parent. Read them from Property > Transform with the door fully open and
    fully closed.
"""

import argparse

from loguru import logger

from telekinesis.medulla.machines import isaacsim


def main(
    prim_path: str,
    usd: str,
    open_position: list[float],
    closed_position: list[float],
) -> None:
    """Load a CNC machine from a USD asset and connect to its door."""

    # ===================== Create Machine =====================================
    machine = isaacsim.CNCMachine(name="my_simulated_cnc_machine")

    try:
        # ==================== Run Skill =======================================
        machine.set_usd(usd)

        # ===================== Connect Machine ================================
        machine.connect(
            simulation_prim_path=prim_path,
            open_position=open_position,
            closed_position=closed_position,
        )
        logger.success(f"Connected to CNC machine door at {prim_path}.")
    except (ConnectionError, RuntimeError, TypeError, ValueError) as e:
        logger.error(f"Error occurred: {e}")
    finally:
        machine.disconnect()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Set a CNC machine USD asset")
    p.add_argument(
        "--prim_path",
        type=str,
        default="/World/cnc_machine/E_body_1/door",
        help="Isaac Sim CNC machine door prim path",
    )
    p.add_argument(
        "--usd",
        type=str,
        default="https://assets.telekinesis.ai/usd/machines/cnc_machine.zip",
        help="HTTP(S) bundle URL or local .usd, directory, or .zip path",
    )
    p.add_argument(
        "--open_position",
        type=float,
        nargs=3,
        default=[-0.68654, -0.05313, 1.208],
        metavar=("X", "Y", "Z"),
        help="Open door position in meters relative to the door parent",
    )
    p.add_argument(
        "--closed_position",
        type=float,
        nargs=3,
        default=[-0.2193, -0.05313, 1.208],
        metavar=("X", "Y", "Z"),
        help="Closed door position in meters relative to the door parent",
    )
    args = p.parse_args()

    main(
        prim_path=args.prim_path,
        usd=args.usd,
        open_position=args.open_position,
        closed_position=args.closed_position,
    )
