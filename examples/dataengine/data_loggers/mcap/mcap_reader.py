"""
Read back an MCAP log and inspect its contents.

Works for any MCAP written by MCAPLogger.
"""

import argparse
import pathlib
from collections import Counter

from telekinesis.dataengine import MCAPLogger


def main(path: pathlib.Path) -> None:
    # ── Generic read (any MCAPLogger file) ───────────────────────────────────
    counts: Counter = Counter()
    first_per_topic: dict = {}

    for topic, obj in MCAPLogger.read(path):
        counts[topic] += 1
        if topic not in first_per_topic:
            first_per_topic[topic] = obj

    print(f"Decoded {sum(counts.values())} messages from {path}")
    for topic, n in counts.items():
        print(f"  {topic}: {n} messages | sample: {first_per_topic[topic]!r}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path",
        type=pathlib.Path,
        help="Path to the MCAP file to read",
    )
    args = parser.parse_args()
    main(args.path)
