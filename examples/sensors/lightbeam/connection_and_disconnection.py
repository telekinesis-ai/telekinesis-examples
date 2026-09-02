"""
Demonstrates connecting to and disconnecting from a lightbeam sensor in a
running Isaac Sim stage.

Supports Isaac Sim only.

Usage:
    python connection_and_disconnection.py --prim_path <PRIM_PATH>

Note:
    Open Isaac Sim and add a lightbeam sensor prim before running this. If
    there is none in the stage yet, follow
    https://docs.isaacsim.omniverse.nvidia.com/5.1.0/sensors/isaacsim_sensors_physx_lightbeam.html
    to create one.
"""

import argparse

from loguru import logger

from telekinesis.medulla.sensors import isaacsim


def main(prim_path: str) -> None:
    """Connects to a lightbeam sensor, then disconnects."""

    #===================== Create Sensor ======================================
    sensor = isaacsim.IsaacSimLightBeamSensor(name="my_simulated_lightbeam")

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
    p.add_argument("--prim_path", type=str, default="/World/LightBeam_Sensor",
                   help='Isaac Sim lightbeam sensor prim path, e.g. '
                        '"/World/LightBeam_Sensor"')
    args = p.parse_args()

    main(prim_path=args.prim_path)
