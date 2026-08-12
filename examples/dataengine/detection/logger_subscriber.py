"""
Detection-logger subscriber example (BabyROS).

Listens on a topic for published frames and writes them to disk as a detection
dataset (YOLO or RF-DETR). Pair it with ``logger_publisher.py``, which publishes
synthetic frames. Start this subscriber first, then run the publisher in a
second terminal. Press Ctrl+C to stop and flush the dataset.

Usage:
    python logger_subscriber.py
"""

import argparse
import pathlib
import time

from telekinesis.dataengine import DetectionLogger, DetectionLoggerSubscriber


def main(
    output_path: pathlib.Path,
    output_format: str = "rfdetr",
    mode: str = "create",
    topic: str = "detection_logger/frames",
) -> None:
    dataset_dir = output_path.resolve()
    dataset_logger = DetectionLogger.create(output_format, dataset_dir, mode=mode)

    print(f"Subscriber: listening on '{topic}' -> {dataset_dir}")
    print("Press Ctrl+C to stop and flush the dataset …")

    # Frames arrive on a background thread; the main thread just waits for Ctrl+C.
    # Leaving the ``with`` block drains the write queue and closes the logger.
    with DetectionLoggerSubscriber(topic=topic, logger=dataset_logger) as sub:
        try:
            while True:
                time.sleep(0.5)
        except KeyboardInterrupt:
            print("\nStopping — draining the write queue …")

    print(f"Dataset written to {dataset_dir}")
    print(f"  written={sub.written} dropped={sub.dropped}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="BabyROS detection-logger subscriber example."
    )
    parser.add_argument(
        "--output-path",
        type=pathlib.Path,
        default="results/subscriber_dataset",
        help="Dataset directory (default: results/subscriber_dataset)",
    )
    parser.add_argument(
        "--format",
        dest="output_format",
        choices=("yolo", "rfdetr"),
        default="yolo",
        help="Dataset format to write (default: yolo)",
    )
    parser.add_argument(
        "--mode",
        choices=("create", "overwrite", "append"),
        default="create",
        help="How to handle an existing non-empty dataset directory "
        "(default: create)",
    )
    args = parser.parse_args()

    main(
        output_path=args.output_path,
        output_format=args.output_format,
        mode=args.mode,
    )
