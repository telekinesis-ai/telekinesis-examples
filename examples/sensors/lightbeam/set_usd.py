"""Load a lightbeam sensor into an Isaac Sim stage from a USD asset.

Supports Isaac Sim only.

Usage:
    python set_usd.py --prim_path <PRIM_PATH>
    python set_usd.py --prim_path <PRIM_PATH> --usd <BUNDLE_URL_OR_PATH>

Note:
    Open Isaac Sim before running this. The asset is loaded only when the
    sensor prim is missing from the open stage. An existing prim is used as
    it is and the configured asset is ignored.
"""

import argparse

from loguru import logger

from telekinesis.medulla.sensors import isaacsim


def main(prim_path: str, usd: str) -> None:
    """Load a lightbeam sensor from a USD asset and connect to it."""

    # ===================== Create Sensor ======================================
    sensor = isaacsim.LightBeamSensor(name="my_simulated_lightbeam")

    try:
        # ==================== Run Skill =======================================
        sensor.set_usd(usd)

        # ===================== Connect Sensor =================================
        sensor.connect(simulation_prim_path=prim_path)
        logger.success(f"Connected to lightbeam sensor at {prim_path}.")
    except (ConnectionError, RuntimeError, TypeError) as e:
        logger.error(f"Error occurred: {e}")
    finally:
        sensor.disconnect()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Set a lightbeam sensor USD asset")
    p.add_argument(
        "--prim_path",
        type=str,
        default="/World/simple_light_beam_sensor/LightBeam_Sensor",
        help="Isaac Sim lightbeam sensor prim path",
    )
    p.add_argument(
        "--usd",
        type=str,
        default=(
            "https://assets.telekinesis.ai/usd/sensors/"
            "simple_light_beam_sensor.zip"
        ),
        help="HTTP(S) bundle URL or local .usd, directory, or .zip path",
    )
    args = p.parse_args()

    main(prim_path=args.prim_path, usd=args.usd)
