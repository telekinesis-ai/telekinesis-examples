"""
Run ALL example scripts in the examples/ folder.

Each example is executed in a fresh Python process to avoid
global state issues (e.g. rerun, gRPC, visualization).

Usage:
    python run_all_examples.py
    python run_all_examples.py --examples_dir path/to/examples
"""

import argparse
import pathlib
import subprocess
import sys
from loguru import logger


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run all example scripts in the examples directory"
    )
    parser.add_argument(
        "--examples_dir",
        type=pathlib.Path,
        default=None,
        help="Directory containing example scripts (defaults to this file's directory)",
    )
    return parser.parse_args()


def run_example_in_subprocess(example_file: pathlib.Path) -> bool:
    """
    Run a single example script in a fresh Python process.

    Returns True if successful, False otherwise.
    """
    cmd = [sys.executable, str(example_file)]

    logger.info(f"Executing: {' '.join(cmd)}")

    result = subprocess.run(cmd)

    return result.returncode == 0


def main():
    args = parse_args()
    examples_dir = args.examples_dir

    if examples_dir is None:
        examples_dir = pathlib.Path(__file__).parent

    this_file = pathlib.Path(__file__).resolve()
    example_files = sorted(f for f in examples_dir.glob("*_example.py") if f.resolve() != this_file)

    if not example_files:
        logger.error(f"No example scripts found in: {examples_dir}")
        sys.exit(1)

    logger.info(f"Found {len(example_files)} examples in: {examples_dir}")

    failed = []
    passed = []

    for idx, example_file in enumerate(example_files, start=1):
        example_name = example_file.stem

        logger.info("=" * 80)
        logger.info(f"[{idx}/{len(example_files)}] Running example: {example_name}")

        ok = run_example_in_subprocess(example_file)

        if ok:
            passed.append(example_name)
            logger.success(f"Completed: {example_name} ({len(passed)}/{len(example_files)} passed)")
        else:
            logger.error(f"FAILED: {example_name} ({len(passed)}/{len(example_files)} passed)")
            failed.append(example_name)

    logger.info("=" * 80)

    if failed:
        logger.warning(
            f"Examples passed: {len(passed)}/{len(example_files)}. Examples failed: {len(failed)}/{len(example_files)}"
        )
        for name in failed:
            logger.warning(f"  - {name}")
        sys.exit(1)

    logger.success(
        f"All examples completed successfully 🎉 ({len(passed)}/{len(example_files)} passed)"
    )
    sys.exit(0)


if __name__ == "__main__":
    main()
