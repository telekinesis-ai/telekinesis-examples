"""
Detection-logger publisher example using BabyROS.

Publishes synthetic frames on a topic as if a detection loop were running.
Pair this with ``logger_subscriber.py``, which receives the frames and writes
a dataset to disk.

Start the subscriber first, then run this publisher in a second terminal.

No input data is required. The frames in this example are random noise.

Usage:
    python logger_publisher.py
"""

import numpy as np

from telekinesis.dataengine.data_loggers import DetectionLoggerPublisher


TOPIC = "detection_logger/frames"
NUM_FRAMES = 20
IMAGE_HEIGHT = 480
IMAGE_WIDTH = 640


def logger_publisher_example() -> None:
    """Publish synthetic frames to the detection logger topic."""
    print(f"Publisher: publishing {NUM_FRAMES} frames to '{TOPIC}'")

    with DetectionLoggerPublisher(
        topic=TOPIC,
        color_order="rgb",
    ) as publisher:
        for frame_index in range(NUM_FRAMES):
            random = np.random.default_rng(frame_index)

            image = random.integers(
                0,
                255,
                (IMAGE_HEIGHT, IMAGE_WIDTH, 3),
                dtype=np.uint8,
            )

            # In a real pipeline, annotations would come from an object detector.
            # ``None`` logs the frame as a background / negative sample.
            annotations = None

            publisher.publish(
                image,
                annotations,
                file_name=f"frame_{frame_index:04d}",
            )

            print(
                f"  [{frame_index + 1}/{NUM_FRAMES}] "
                f"published frame_{frame_index:04d}"
            )

    print("Publisher done.")


if __name__ == "__main__":
    logger_publisher_example()