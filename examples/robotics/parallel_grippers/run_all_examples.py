"""
Run ALL parallel gripper example scripts in this folder, one after the other.

Each example is executed in a fresh Python process to avoid global state
issues (e.g. rerun, serial/socket handles, visualization). Each connection
argument is forwarded only to the examples that declare it — the model-only
examples run on the kinematic model, take no arguments at all, and so are run
bare.

Examples run in a hardware-safe order: the model-only examples first, then the
gripper is activated, defaults are configured, and motion skills are exercised
last. Any example not listed in RUN_ORDER is appended alphabetically.

Note: the visualization examples spawn a Rerun viewer window and block until
it is closed.

Usage:
    python run_all_examples.py --ip <GRIPPER_IP>
    python run_all_examples.py --protocol MODBUS_RTU --serial-port COM4
    python run_all_examples.py --prim_path <PRIM_PATH>
"""

import argparse
import pathlib
import subprocess
import sys
import time
from loguru import logger

# Pause between examples, giving the gripper time to settle and the
# previous process time to release its connection.
DELAY_BETWEEN_EXAMPLES_S = 2

# Hardware-safe execution order, relative to this file's directory. Files not
# listed here run afterwards, in alphabetical order.
RUN_ORDER = [
    "get_visual_meshes_data",
    "get_link_transforms",
    "get_visual_mesh_transforms",
    "visualize_rerun",
    "connection_and_disconnection",
    "activate",
    "set_unit",
    "set_position_range",
    "set_speed",
    "set_force",
    "get_current_position",
    "open",
    "close",
    "move",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run all parallel gripper example scripts in this directory"
    )
    parser.add_argument(
        "--examples_dir",
        type=pathlib.Path,
        default=None,
        help="Directory containing example scripts (defaults to this file's directory)",
    )
    parser.add_argument("--protocol",
                        choices=["URCAP", "MODBUS_RTU"],
                        default="URCAP")
    parser.add_argument("--ip", default=None, help="IP for Robotiq Gripper")
    parser.add_argument("--serial-port", dest="serial_port", default="COM4",
                        help="Serial port for MODBUS_RTU")
    parser.add_argument("--prim_path", type=str, default=None,
                        help='Isaac Sim gripper prim path, e.g. "/World/robotiq_2f85"')
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


def supported_args(example_file: pathlib.Path,
                   connection_args: list[tuple[str, str]]) -> list[str]:
    """
    Select the connection arguments an example actually declares.

    The model-only examples take no arguments, so passing them any flag would
    make argparse exit with an error.
    """
    source = example_file.read_text(encoding="utf-8")

    selected = []
    for flag, value in connection_args:
        if flag in source:
            selected += [flag, value]
    return selected


def run_example_in_subprocess(example_file: pathlib.Path,
                              connection_args: list[tuple[str, str]]) -> bool:
    """
    Run a single example script in a fresh Python process.

    Returns True if successful, False otherwise.
    """
    cmd = [sys.executable, str(example_file),
           *supported_args(example_file, connection_args)]

    logger.info(f"Executing: {' '.join(cmd)}")

    result = subprocess.run(cmd)

    return result.returncode == 0


def main():
    args = parse_args()
    examples_dir = args.examples_dir

    if examples_dir is None:
        examples_dir = pathlib.Path(__file__).parent

    connection_args = [("--protocol", args.protocol),
                       ("--serial-port", args.serial_port)]
    if args.ip is not None:
        connection_args.append(("--ip", args.ip))
    if args.prim_path is not None:
        connection_args.append(("--prim_path", args.prim_path))

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
