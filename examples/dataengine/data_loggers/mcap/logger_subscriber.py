"""
MCAP Logger Example

Subscribes to every publisher on the Zenoh network (wildcard "**") and
streams all messages into an MCAP file. Run any number of BabyROS
publishers in other terminals and this will discover and log them
automatically.
"""

import argparse
import pathlib
import time

from telekinesis.dataengine import MCAPLogger


def logger_subscriber_example(path: pathlib.Path) -> None:
    logger = MCAPLogger(path)
    logger.start()

    print(f"Logging all topics to {path} ... (Press Ctrl+C to stop)")

    try:
        while True:
            time.sleep(1)
            print(
                f"  {logger.num_messages} messages "
                f"from topics: {logger.topics}"
            )

    except KeyboardInterrupt:
        print("\n[MCAPLogger] Interrupted by user.")

    finally:
        logger.stop()
        print(f"[MCAPLogger] Saved {path}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "path",
        nargs="?",
        type=pathlib.Path,
        default=pathlib.Path("./telekinesis_log.mcap"),
        help="Path to the MCAP file to write (default: %(default)s).",
    )

    args = parser.parse_args()

    logger_subscriber_example(args.path)
