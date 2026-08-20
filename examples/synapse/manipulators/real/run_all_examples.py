"""
Run ALL real-hardware manipulator example scripts in this folder, one after
the other.

Each example is executed in a fresh Python process to avoid global state
issues (e.g. rerun, RTDE handles, visualization). The ``--ip`` argument is
forwarded to every example.

Examples run in a hardware-safe order: connection is checked first, then
read-only status/state getters, then TCP and tool setup, then motion
skills, then the contact-detection / stop / protective-stop skills that
leave the robot halted or requiring pendant acknowledgement. Any example not
listed in RUN_ORDER is appended alphabetically. ``ur10e_virtual_controller``
always targets a local URSim instance regardless of ``--ip`` and is left to
run last.

By default, the name of the next example is printed and execution pauses
until Enter is pressed, so you can watch each one individually and keep the
real robot supervised. Pass ``--non-interactive`` to run straight through
instead.

Usage:
    python run_all_examples.py --ip <ROBOT_IP> [--non-interactive]
"""

import argparse
import pathlib
import subprocess
import sys
import time
from loguru import logger

# Pause between examples, giving the robot time to settle and the previous
# process time to release its RTDE connection.
DELAY_BETWEEN_EXAMPLES_S = 2

# Hardware-safe execution order. Files not listed here run afterwards,
# in alphabetical order.
RUN_ORDER = [
    "connection_and_disconnection",
    "is_connected",
    "get_robot_mode",
    "get_robot_status",
    "get_runtime_state",
    "get_safety_mode",
    "get_safety_status_bits",
    "is_protective_stopped",
    "is_emergency_stopped",
    "is_program_running_on_controller",
    "is_steady",
    "get_target_speed_fraction",
    "get_speed_scaling_combined",
    "get_controller_frequency",
    "get_publisher_names_and_types",
    "get_publisher_hz",
    "get_timestamp",
    "get_state",
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
    "in_joint_limits",
    "is_pose_within_safety_limits",
    "is_joints_within_safety_limits",
    "get_tcps",
    "add_tcp",
    "update_tcp",
    "change_active_tcp",
    "set_controller_interface_tcp_as_active",
    "delete_tcp",
    "attach_tool",
    "detach_tool",
    "set_joint_positions",
    "set_cartesian_pose",
    "set_cartesian_pose_in_joint_space",
    "set_joint_position_in_cartesian_space",
    "servo_joint",
    "servo_cartesian",
    "servo_circular",
    "servo_stop",
    "stop_joint_motion",
    "stop_cartesian_motion",
    "start_and_stop_jog_mode",
    "start_and_stop_freedrive_mode",
    # "start_and_stop_teach_mode",
    "is_tool_in_contact",
    "move_until_contact",
    "contact_detection",
    "trigger_protective_stop",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run all real-hardware manipulator example scripts in this directory"
    )
    parser.add_argument(
        "--examples_dir",
        type=pathlib.Path,
        default=None,
        help="Directory containing example scripts (defaults to this file's directory)",
    )
    parser.add_argument("--ip", default=None, help="UR robot IP address")
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
    if args.ip is not None:
        connection_args += ["--ip", args.ip]

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
