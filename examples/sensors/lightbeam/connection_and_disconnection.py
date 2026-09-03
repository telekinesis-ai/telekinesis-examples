"""
Demonstrates connecting to and disconnecting from a lightbeam sensor in a
running Isaac Sim stage.

Supports Isaac Sim only.

Usage:
    python connection_and_disconnection.py --prim_path <PRIM_PATH>
    python connection_and_disconnection.py --prim_path <PRIM_PATH> --load_usd

Note:
    Open Isaac Sim and add a lightbeam sensor prim before running this. If
    there is none in the stage yet, follow
    https://docs.isaacsim.omniverse.nvidia.com/5.1.0/sensors/isaacsim_sensors_physx_lightbeam.html
    to create one, or pass --load_usd to add one to the open stage -- this
    keeps whatever is already in the stage.
"""

import argparse

from loguru import logger

from telekinesis.medulla.sensors import isaacsim


def main(prim_path: str, load_usd: bool) -> None:
    """Connects to a lightbeam sensor, then disconnects."""

    if load_usd:
        # ===================== Load Demo Scene (Optional) ===========================
        from telekinesis import datatypes, isaacsim_client

        client = isaacsim_client.IsaacSimClient(
            api_key="",
            base_url="http://127.0.0.1:8766",
            websocket_base_url="ws://127.0.0.1:8766",
        )
        asset = datatypes.USD.from_url(
            "https://assets.telekinesis.ai/usd/sensors/simple_light_beam_sensor.zip"
        )
        client.stage.add_to_scene(uri=asset.path.as_posix(),
                                  prim_path="/World/simple_light_beam_sensor")

    # ===================== Create Sensor ======================================
    sensor = isaacsim.LightBeamSensor(name="my_simulated_lightbeam")

    try:
        # ==================== Run Skill ============================================
        sensor.connect(simulation_prim_path=prim_path)
        logger.success(f"Connected: {sensor.is_connected}.")
    except (ConnectionError, RuntimeError) as e:
        logger.error(f"Error occurred: {e}")
    finally:
        sensor.disconnect()


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Connect to and disconnect from a lightbeam sensor in Isaac Sim")
    p.add_argument(
        "--prim_path",
        type=str,
        default="/World/simple_light_beam_sensor/LightBeam_Sensor",
        help='Isaac Sim lightbeam sensor prim path, e.g. '
        '"/World/simple_light_beam_sensor/LightBeam_Sensor"')
    p.add_argument("--load_usd", action=argparse.BooleanOptionalAction, default=False,
                   help="Add the bundled demo lightbeam sensor to the open "
                        "stage at /World/simple_light_beam_sensor before "
                        "connecting. Use this if you don't already have one "
                        "in the stage.")
    args = p.parse_args()

    main(prim_path=args.prim_path, load_usd=args.load_usd)
