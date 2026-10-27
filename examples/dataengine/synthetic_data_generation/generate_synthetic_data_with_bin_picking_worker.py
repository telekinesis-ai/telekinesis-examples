"""Generate a bin-picking dataset using a YAML configuration."""

import argparse
from collections.abc import Sequence
import os
from pathlib import Path
import sys


def main(argv: Sequence[str] = ()) -> None:
    """Generate and merge a dataset using the selected YAML configuration.

    Args:
        argv: CLI arguments. Defaults to the example YAML with preview disabled.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config-file",
        type=Path,
        default=(
            Path(__file__).resolve().parents[0]
            / "bin_picking_example.yaml"
        ),
        help="Path to the bin-picking YAML configuration (default: %(default)s).",
    )
    parser.add_argument(
        "--preview",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Open the generated dataset in the interactive viewer.",
    )
    args = parser.parse_args(argv)

    os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

    from telekinesis.illusion.workers.bin_picking_worker import BinPickingWorker

    worker = BinPickingWorker(args.config_file)
    worker.generate()
    worker.merge_shards(preview=args.preview)


if __name__ == "__main__":
    main(sys.argv[1:])
