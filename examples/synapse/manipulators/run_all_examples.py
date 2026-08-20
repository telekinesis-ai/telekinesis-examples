"""
Run ALL kinematics/trajectory manipulator example scripts in this folder
(not the real\\ or virtual\\ subfolders), one after the other.

Each example is executed in a fresh Python process to avoid global state
issues (e.g. rerun). None of these examples connect to hardware or
simulation, so no connection arguments are needed.

Any example not listed in RUN_ORDER is appended alphabetically.

Usage:
    python run_all_examples.py
"""

import pathlib
import subprocess
import sys
import time
import argparse
from loguru import logger

# Pause between examples, giving the previous process time to release its
# rerun handles.
DELAY_BETWEEN_EXAMPLES_S = 2

# Execution order. Files not listed here run afterwards, in alphabetical order.
RUN_ORDER = [
    "forward_kinematics",
    "inverse_kinematics",
    "setup_kinematics_solver",
    "set_default_joint_configuration",
    "joint_trajectory_generator_example",
    "cartesian_trajectory_generator_example",
    "joint_trajectory_controller",
    "visualize",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run all kinematics/trajectory manipulator example scripts in this directory"
    )
    parser.add_argument(
        "--examples_dir",
        type=pathlib.Path,
        default=None,
        help="Directory containing example scripts (defaults to this file's directory)",
    )
    return parser.parse_args()


def collect_examples(examples_dir: pathlib.Path) -> list[pathlib.Path]:
    """
    Collect runnable example scripts, ordered by RUN_ORDER.

    This runner itself, private modules and empty placeholder files are skipped.
    The real\\ and virtual\\ subfolders (which have their own runners) are skipped.
    """
    this_file = pathlib.Path(__file__).resolve()

    candidates = {}
    for f in examples_dir.glob("*.py"):
        if f.resolve() == this_file or f.name.startswith("_"):
            continue
        if f.stat().st_size == 0:
            logger.warning(f"Skipping empty example: {f.name}")
            continue
        candidates[f.stem] = f

    ordered = [candidates.pop(name) for name in RUN_ORDER if name in candidates]
    ordered.extend(candidates[name] for name in sorted(candidates))
    return ordered


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

    example_files = collect_examples(examples_dir)

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

        time.sleep(DELAY_BETWEEN_EXAMPLES_S)

    logger.info("=" * 80)

    if failed:
        logger.warning(
            f"Examples passed: {len(passed)}/{len(example_files)}. "
            f"Examples failed: {len(failed)}/{len(example_files)}"
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
