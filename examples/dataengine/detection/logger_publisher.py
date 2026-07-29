"""
Detection-logger publisher example (BabyROS).

Publishes synthetic frames on a topic as if a detection loop were running. Pair
it with ``logger_subscriber.py``, which receives the frames and writes a dataset
to disk. Start the subscriber first, then run this publisher in a second
terminal.

No data required — the frames here are random noise.

Usage:
    python logger_publisher.py
"""

import argparse

import numpy as np

from telekinesis.dataengine import DetectionLoggerPublisher


def main(
    topic: str = "detection_logger/frames",
    num_frames: int = 20,
) -> None:
    print(f"Publisher: publishing {num_frames} frames to '{topic}' …")
    with DetectionLoggerPublisher(topic=topic, color_order="rgb") as pub:
        for frame_index in range(num_frames):
            random = np.random.default_rng(frame_index)
            rgb = random.integers(0, 255, (480, 640, 3), dtype=np.uint8)

            # In a real pipeline, detections come from an object detector here.
            # This example publishes each frame with no annotations (None) so
            # the subscriber logs them as background / negative samples.
            annotations = None

            pub.publish(rgb, annotations, file_name=f"frame_{frame_index:04d}")
            print(f"  [{frame_index + 1}/{num_frames}] sent")

    print("Publisher done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="BabyROS detection-logger publisher example."
    )
    parser.add_argument(
        "--topic",
        default="detection_logger/frames",
        help="Topic to publish on (default: detection_logger/frames)",
    )
    args = parser.parse_args()

    main(topic=args.topic)
