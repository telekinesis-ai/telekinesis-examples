"""
End-to-end usage flow: publish -> log -> read back -> summarize.

Runs the full MCAP pipeline in one script so you can see how the other
examples in this directory connect. Each stage below has a standalone
counterpart with its own CLI flags:

  1. Publish three BabyROS topics at different rates (imu, camera,
     joint_states) -- see dummy_publishers.py.
  2. Subscribe to every topic ("**") and log them into one MCAP file --
     see logger_subscriber.py.
  3. Read the MCAP file back and print per-topic counts/samples -- see
     mcap_reader.py.

Everything is written to --output-path, which is overwritten on every run
so the pipeline is safe to re-run as-is.

Usage:
    python tutorial.py
    python tutorial.py --output-path results/tutorial.mcap --duration 5
"""

from __future__ import annotations

import argparse
import pathlib
import threading
import time
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from loguru import logger

import babyros
from telekinesis.dataengine import MCAPLogger


@dataclass
class Stream:
    """One publisher spec: which topic, how fast, and how to build a message."""

    topic: str
    hz: float
    make_message: Callable[[int, np.random.Generator], Any]


def _make_imu(seq: int, rng: np.random.Generator) -> dict:
    """Fake IMU sample -- gravity plus a little noise."""
    return {
        "acceleration": (np.array([0.0, 0.0, 9.81]) + rng.normal(0, 0.02, 3)).tolist(),
        "gyro": rng.normal(0, 0.01, 3).tolist(),
        "seq": [seq],
    }


def _make_camera(seq: int, rng: np.random.Generator) -> dict:
    """Fake camera frame -- random RGB noise."""
    frame = rng.integers(0, 255, (240, 320, 3), dtype=np.uint8)
    return {
        "image": frame,
        "seq": [seq],
    }


def _make_joint_states(seq: int, rng: np.random.Generator) -> dict:
    """Fake joint states -- positions drifting on a slow sinusoid."""
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
    seq = 0
    try:
        next_tick = time.perf_counter()
        while not stop.is_set():
            pub.publish(data=stream.make_message(seq, rng))
            seq += 1
            next_tick += period
            stop.wait(max(0.0, next_tick - time.perf_counter()))
    finally:
        pub.delete()
    return seq


def main(output_path: pathlib.Path, duration: float) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Step 1/3: publish {len(STREAMS)} topics at their native rates")
    stop = threading.Event()
    threads = [
        threading.Thread(target=_run_stream, args=(stream, stop, seed), name=stream.topic)
        for seed, stream in enumerate(STREAMS)
    ]
    for thread in threads:
        thread.start()
    for stream in STREAMS:
        logger.info(f"publishing '{stream.topic}' at {stream.hz:g} Hz")

    print(f"\nStep 2/3: subscribe to every topic ('**') and log to {output_path}")
    mcap_logger = MCAPLogger(output_path)
    try:
        time.sleep(duration)
    finally:
        stop.set()
        for thread in threads:
            thread.join()
        logger.success(
            f"logged {mcap_logger.num_messages()} messages from topics: {mcap_logger.topics()}"
        )
        mcap_logger.delete()

    print(f"\nStep 3/3: read {output_path} back and summarize")
    counts: Counter = Counter()
    first_per_topic: dict = {}
    for topic, obj in MCAPLogger.read(output_path):
        counts[topic] += 1
        if topic not in first_per_topic:
            first_per_topic[topic] = obj

    print(f"Decoded {sum(counts.values())} messages from {output_path}")
    for topic, n in counts.items():
        print(f"  {topic}: {n} messages | sample: {first_per_topic[topic]!r}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the publish -> log -> read -> summarize MCAP pipeline."
    )
    parser.add_argument(
        "--output-path",
        type=pathlib.Path,
        default=pathlib.Path("results/tutorial.mcap"),
        help="Path to the MCAP file to write (default: %(default)s)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=3.0,
        help="Seconds to publish/record before stopping (default: %(default)s)",
    )
    args = parser.parse_args()

    main(output_path=args.output_path, duration=args.duration)
