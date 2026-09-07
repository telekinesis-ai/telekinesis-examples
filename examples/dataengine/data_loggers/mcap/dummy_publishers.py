"""
Multi-rate publisher example (pairs with ``logger_subscriber.py``).

Spins up three independent BabyROS publishers, each on its own thread and its
own topic, publishing at a different frequency:

  - ``imu``          →  100 Hz   (dict of accel / gyro / seq)
  - ``camera``       →   30 Hz   (a synthetic ``Image`` frame)
  - ``joint_states`` →    5 Hz   (dict of joint positions / velocities)

Run the MCAP logger first, then this publisher, in two terminals:

    python examples/data-engine/mcap/logger_subscriber.py
    python examples/data-engine/mcap/dummy_publishers.py

The logger subscribes to everything ("**"), so it discovers all three topics
automatically and records them at their native rates. Press Ctrl+C here to stop.
"""

import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from loguru import logger

import babyros


@dataclass
class Stream:
    """One publisher spec: which topic, how fast, and how to build a message."""

    topic: str
    hz: float
    make_message: Callable[[int, np.random.Generator], Any]


def _make_imu(seq: int, rng: np.random.Generator) -> dict:
    """Fake IMU sample — gravity plus a little noise."""
    return {
        "acceleration": (np.array([0.0, 0.0, 9.81]) + rng.normal(0, 0.02, 3)).tolist(),
        "gyro": rng.normal(0, 0.01, 3).tolist(),
        "seq": [seq],
    }


def _make_camera(seq: int, rng: np.random.Generator) -> dict:
    """Fake camera frame — random RGB noise."""
    frame = rng.integers(0, 255, (240, 320, 3), dtype=np.uint8)
    return {
        "image": frame,
        "seq": [seq],
    }


def _make_joint_states(seq: int, rng: np.random.Generator) -> dict:
    """Fake joint states — positions drifting on a slow sinusoid."""
    phase = seq * 0.1
    return {
        "position": (np.sin(phase + np.arange(7))).tolist(),
        "velocity": rng.normal(0, 0.05, 7).tolist(),
        "seq": [seq],
    }


STREAMS = [
    Stream(topic="imu", hz=100.0, make_message=_make_imu),
    Stream(topic="camera", hz=30.0, make_message=_make_camera),
    Stream(topic="joint_states", hz=5.0, make_message=_make_joint_states),
]


def _run_stream(stream: Stream, stop: threading.Event, seed: int) -> None:
    """Publish ``stream`` at its target rate until ``stop`` is set."""
    rng = np.random.default_rng(seed)
    period = 1.0 / stream.hz
    pub = babyros.node.Publisher(topic=stream.topic)
    logger.info(f"publishing '{stream.topic}' at {stream.hz:g} Hz")
    seq = 0
    try:
        next_tick = time.perf_counter()
        while not stop.is_set():
            pub.publish(data=stream.make_message(seq, rng))
            seq += 1
            next_tick += period
            # Sleep until the next tick; if we fell behind, fire immediately.
            stop.wait(max(0.0, next_tick - time.perf_counter()))
    finally:
        pub.delete()
        logger.info(f"'{stream.topic}' stopped after {seq} messages")


def main() -> None:
    stop = threading.Event()
    threads = [
        threading.Thread(
            target=_run_stream, args=(stream, stop, seed), name=stream.topic, daemon=True
        )
        for seed, stream in enumerate(STREAMS)
    ]
    for thread in threads:
        thread.start()

    print("Publishing on:", ", ".join(f"{s.topic}@{s.hz:g}Hz" for s in STREAMS))
    print("Press Ctrl+C to stop …")
    try:
        while any(thread.is_alive() for thread in threads):
            time.sleep(0.2)
    except KeyboardInterrupt:
        print("\nStopping publishers …")
    finally:
        stop.set()
        for thread in threads:
            thread.join()
    print("Done.")


if __name__ == "__main__":
    main()
