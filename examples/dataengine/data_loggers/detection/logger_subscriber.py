"""
Detection-logger subscriber example using BabyROS.

Listens on a topic for published frames and writes them to disk as a detection
dataset in YOLO or RF-DETR format.

Pair this with `logger_publisher.py`, which publishes synthetic frames.

Start this subscriber first, then run the publisher in a second terminal.
Press Ctrl+C to stop and flush the dataset.

Usage:
    python logger_subscriber.py
"""

from pathlib import Path
import time

from telekinesis.dataengine.data_loggers import DetectionLogger, DetectionLoggerSubscriber


OUTPUT_PATH = Path("results/subscriber_dataset")
OUTPUT_FORMAT = "yolo"
MODE = "overwrite"
TOPIC = "detection_logger/frames"


def logger_subscriber_example() -> None:
    """Listen for detection frames and write them to a dataset."""
    dataset_dir = OUTPUT_PATH.resolve()
    dataset_logger = DetectionLogger.create(
        OUTPUT_FORMAT,
        dataset_dir,
        mode=MODE,
    )

    print(f"Subscriber: listening on '{TOPIC}' -> {dataset_dir}")
    print("Press Ctrl+C to stop and flush the dataset.")

    with DetectionLoggerSubscriber(
        topic=TOPIC,
        logger=dataset_logger,
    ) as subscriber:
        try:
            while True:
                time.sleep(0.5)
        except KeyboardInterrupt:
            print("\nStopping — draining the write queue...")

    print(f"Dataset written to {dataset_dir}")
    print(
        f"  written={subscriber.written} "
        f"dropped={subscriber.dropped}"
    )


if __name__ == "__main__":
    logger_subscriber_example()