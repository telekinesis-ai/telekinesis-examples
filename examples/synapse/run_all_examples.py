"""Smoke-test runner for all synapse example groups.

Runs every example with try/except so absent hardware (RTDE ConnectionError) and
skipped motion (SYNAPSE_EXAMPLES_NONINTERACTIVE=1) are reported but do not abort
the suite.

    python run_all_examples.py                  # run all groups
    python run_all_examples.py --group motion   # run a single group
    python run_all_examples.py --ip 192.168.1.5 # custom IP
"""

import argparse
import importlib
import os
import sys

from loguru import logger


GROUP_MODULES = [
    "kinematics_examples",
    "connection_examples",
    "motion_examples",
    "servo_control_examples",
    "speed_control_examples",
    "force_control_examples",
    "state_reading_examples",
    "robot_status_examples",
    "joint_telemetry_examples",
    "tcp_telemetry_examples",
    "io_and_configuration_examples",
    "safety_checks_examples",
    "contact_detection_examples",
    "ft_sensor_setup_examples",
    "dynamics_examples",
    "target_state_examples",
    "electrical_examples",
    "diagnostics_examples",
    "joint_detail_examples",
    "custom_scripting_examples",
    "watchdog_examples",
    "recording_examples",
    "tools_examples",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Synapse examples smoke-test runner")
    parser.add_argument("--ip", type=str, default="192.168.1.2", help="UR robot IP")
    parser.add_argument(
        "--group",
        type=str,
        help="Single group to run (e.g. 'motion', 'kinematics'). If omitted, runs all.",
    )
    parser.add_argument(
        "--interactive-motion",
        action="store_true",
        help="Allow real motion examples to prompt for 'yes' (default: skip motion via SYNAPSE_EXAMPLES_NONINTERACTIVE=1).",
    )
    return parser.parse_args()


def _resolve_group_module(name: str):
    """Match 'motion' -> 'motion_examples', or accept full module name."""
    if name in GROUP_MODULES:
        return name
    candidate = f"{name}_examples"
    if candidate in GROUP_MODULES:
        return candidate
    raise SystemExit(
        f"Unknown group '{name}'. Available: "
        + ", ".join(m.removesuffix("_examples") for m in GROUP_MODULES)
    )


def main():
    args = parse_args()

    if not args.interactive_motion:
        os.environ["SYNAPSE_EXAMPLES_NONINTERACTIVE"] = "1"

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    if args.group:
        modules = [_resolve_group_module(args.group)]
    else:
        modules = GROUP_MODULES

    results = []  # (group, example, status, detail)
    for mod_name in modules:
        try:
            mod = importlib.import_module(mod_name)
        except Exception as e:
            logger.error(f"[{mod_name}] FAILED TO IMPORT: {type(e).__name__}: {e}")
            results.append((mod_name, "<import>", "IMPORT_FAIL", str(e)))
            continue

        examples = mod.get_example_dict(args.ip)
        logger.info(f"=== {mod_name}: {len(examples)} examples ===")
        for name, fn in examples.items():
            try:
                fn()
                results.append((mod_name, name, "OK", ""))
                logger.success(f"[{mod_name}.{name}] OK")
            except Exception as e:
                status = "SKIP" if "NONINTERACTIVE" in str(e) else "FAIL"
                results.append((mod_name, name, status, f"{type(e).__name__}: {e}"))
                logger.warning(f"[{mod_name}.{name}] {status}: {type(e).__name__}: {e}")

    # Summary
    logger.info("=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    by_status = {"OK": 0, "SKIP": 0, "FAIL": 0, "IMPORT_FAIL": 0}
    for _, _, status, _ in results:
        by_status[status] = by_status.get(status, 0) + 1
    for status, count in by_status.items():
        logger.info(f"  {status}: {count}")
    logger.info(f"  total: {len(results)}")

    if by_status.get("FAIL", 0) or by_status.get("IMPORT_FAIL", 0):
        for group, name, status, detail in results:
            if status in ("FAIL", "IMPORT_FAIL"):
                logger.error(f"  {status}: {group}.{name} — {detail}")


if __name__ == "__main__":
    main()
