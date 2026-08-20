"""
Run ALL virtual/offline manipulator example scripts in this folder, one
after the other.

Each example is executed in a fresh Python process to avoid global state
issues (e.g. rerun). The ``--prim`` argument is forwarded to every example;
scripts that don't take it (nearly all of them run purely offline on the
kinematic model) simply ignore it.

By default, the name of the next example is printed and execution pauses
until Enter is pressed, so you can watch each one individually. Pass
``--non-interactive`` to run straight through instead.

Examples run in a settle-friendly order: read-only state getters first,
then TCP and tool setup, then motion skills. Any example not listed in
RUN_ORDER is appended alphabetically.

Connecting/disconnecting is not exercised here: offline mode has no
connection to make. See ``real/connection_and_disconnection.py`` for real
hardware; a simulation-mode equivalent will be added separately.

Usage:
    python run_all_examples.py [--prim <ROBOT_PRIM_PATH>] [--non-interactive]
"""

import argparse
import pathlib
import subprocess
import sys
import time
from loguru import logger

# Pause between examples, giving the previous process time to release its
# rerun/simulation handles.
DELAY_BETWEEN_EXAMPLES_S = 2

# Settle-friendly execution order. Files not listed here run afterwards,
# in alphabetical order.
RUN_ORDER = [
    "is_connected",
    "get_timestamp",
    "get_state",
    "get_cartesian_pose",
    "get_joint_positions",
    "get_joint_velocities",
    "get_joint_torques",
    "get_tcp_speed",
    "get_tcp_force",
    "get_target_joint_positions",
    "get_target_joint_velocities",
    "get_target_joint_accelerations",
    "get_target_tcp_pose",
    "get_target_tcp_speed",
    "get_tcps",
    "add_tcp",
    "update_tcp",
    "change_active_tcp",
    "delete_tcp",
    "attach_tool",
    "set_joint_positions",
    "set_cartesian_pose",
    "set_cartesian_pose_in_joint_space",
    "set_joint_position_in_cartesian_space",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run all virtual/offline manipulator example scripts in this directory"
    )
    parser.add_argument(
        "--examples_dir",
        type=pathlib.Path,
        default=None,
        help="Directory containing example scripts (defaults to this file's directory)",
    )
    parser.add_argument("--prim", default=None, help="UR robot primitive path in isaacsim")
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Run straight through without pausing for Enter before each example",
    )
    return parser.parse_args()


def collect_examples(examples_dir: pathlib.Path) -> list[pathlib.Path]:
    """
    Collect runnable example scripts, ordered by RUN_ORDER.

    This runner itself, private modules and empty placeholder files are skipped.
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


def run_example_in_subprocess(example_file: pathlib.Path,
                              connection_args: list[str]) -> bool:
    """
    Run a single example script in a fresh Python process.

    Returns True if successful, False otherwise.
    """
    cmd = [sys.executable, str(example_file), *connection_args]

    logger.info(f"Executing: {' '.join(cmd)}")

    result = subprocess.run(cmd)

    return result.returncode == 0


def main():
    args = parse_args()
    examples_dir = args.examples_dir

    if examples_dir is None:
        examples_dir = pathlib.Path(__file__).parent

    connection_args = []
    if args.prim is not None:
        connection_args += ["--prim", args.prim]

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
        logger.info(f"[{idx}/{len(example_files)}] Next example: {example_name}")

        if not args.non_interactive:
            input("Press Enter to run it (Ctrl+C to abort)... ")

        logger.info(f"[{idx}/{len(example_files)}] Running example: {example_name}")

        ok = run_example_in_subprocess(example_file, connection_args)

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
