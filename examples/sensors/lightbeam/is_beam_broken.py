"""
Watches a lightbeam sensor in a running Isaac Sim stage until the beam breaks.

Supports Isaac Sim only. The sensor is sampled from the last physics step, so
the simulation timeline must be playing -- a stopped simulation has no
reading to give.

Usage:
    python is_beam_broken.py --prim_path <PRIM_PATH>
    python is_beam_broken.py --prim_path <PRIM_PATH> --watch_seconds 20

Note:
    Open Isaac Sim and add a lightbeam sensor prim before running this. If
    there is none in the stage yet, follow
    https://docs.isaacsim.omniverse.nvidia.com/5.1.0/sensors/isaacsim_sensors_physx_lightbeam.html
    to create one. Place an object with a collider in front of the beam
    before running this to see it detected.
"""

import argparse
import time

from loguru import logger

from telekinesis.medulla.sensors import isaacsim

POLL_SECONDS = 0.1
PRINT_SECONDS = 1.0


def main(prim_path: str, watch_seconds: float) -> None:
    """Watches a lightbeam sensor until its beam breaks or time runs out."""

    #===================== Create Sensor ======================================
    sensor = isaacsim.LightBeamSensor(name="my_simulated_lightbeam")

    try:
        #===================== Connect Sensor ==================================
        sensor.connect(simulation_prim_path=prim_path)

        # ==================== Run Skill ============================================
        logger.info(f"Watching lightbeam sensor {sensor.name}.")
        deadline = time.monotonic() + watch_seconds
        last_print = 0.0
        while time.monotonic() < deadline:
            if sensor.is_beam_broken():
                logger.success("Object detected: beam broken.")
                break
            now = time.monotonic()
            if now - last_print >= PRINT_SECONDS:
                logger.info("No object detected: beam not broken.")
                last_print = now
            time.sleep(POLL_SECONDS)
        else:
            logger.info("Nothing broke the beam.")
    except (ConnectionError, RuntimeError) as e:
        logger.error(f"Error occurred: {e}")
    finally:
        sensor.disconnect()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Watch a lightbeam sensor in Isaac Sim")
    p.add_argument("--prim_path", type=str, default="/World/LightBeam_Sensor",
                   help='Isaac Sim lightbeam sensor prim path, e.g. '
                        '"/World/LightBeam_Sensor"')
    p.add_argument("--watch_seconds", type=float, default=10.0,
                   help="How long to watch the sensor before giving up")
    args = p.parse_args()

    main(prim_path=args.prim_path, watch_seconds=args.watch_seconds)
