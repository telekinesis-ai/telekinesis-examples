"""Stream lightbeam states from Isaac Sim for a fixed duration.

Supports Isaac Sim only. Beam states are published as
``telekinesis.datatypes.Bool`` values on
``medulla/sensor/LightBeamSensor/<name>/stream``.

Usage:
    python start_and_stop_stream.py --prim_path <PRIM_PATH>
    python start_and_stop_stream.py --stream_seconds 10 --frequency_hz 20

Note:
    Open Isaac Sim before running this. A stage that does not hold the sensor
    prim gets the bundled demo sensor added at
    /World/simple_light_beam_sensor.
"""

import argparse
import time

from loguru import logger

from telekinesis.medulla.sensors import isaacsim


def main(prim_path: str, stream_seconds: float, frequency_hz: float) -> None:
    """Publish beam states for the requested duration, then stop."""

    # ===================== Create Sensor ======================================
    sensor = isaacsim.LightBeamSensor(name="my_simulated_lightbeam")
    sensor.set_usd(
        "https://assets.telekinesis.ai/usd/sensors/"
        "simple_light_beam_sensor.zip"
    )

    try:
        # ===================== Connect Sensor =================================
        sensor.connect(simulation_prim_path=prim_path)
        sensor.stream_frequency_hz = frequency_hz

        # ==================== Run Skill =======================================
        sensor.start_stream()
        logger.info(
            f"Streaming at up to {frequency_hz} Hz for {stream_seconds} s."
        )
        time.sleep(stream_seconds)

        sensor.stop_stream()
        logger.success(f"Stopped. Streaming: {sensor.is_streaming}.")
    except (ConnectionError, RuntimeError, TypeError, ValueError) as e:
        logger.error(f"Error occurred: {e}")
    finally:
        sensor.disconnect()


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Start and stop a lightbeam sensor stream in Isaac Sim"
    )
    p.add_argument(
        "--prim_path",
        type=str,
        default="/World/simple_light_beam_sensor/LightBeam_Sensor",
        help="Isaac Sim lightbeam sensor prim path",
    )
    p.add_argument(
        "--stream_seconds",
        type=float,
        default=5.0,
        help="How long to publish beam states",
    )
    p.add_argument(
        "--frequency_hz",
        type=float,
        default=100.0,
        help="Maximum beam-state publication rate in Hz",
    )
    args = p.parse_args()

    main(
        prim_path=args.prim_path,
        stream_seconds=args.stream_seconds,
        frequency_hz=args.frequency_hz,
    )
