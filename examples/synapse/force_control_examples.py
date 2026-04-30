"""Force Control examples for the Synapse SDK on Universal Robots UR10e."""

import argparse
import difflib
import os
import time

from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots



def _confirm_real_motion(description: str) -> None:
    """Print a loud warning and require explicit 'yes' to proceed (bypassed by SYNAPSE_EXAMPLES_NONINTERACTIVE=1)."""
    logger.warning("=" * 70)
    logger.warning("REAL ROBOT MOTION ABOUT TO EXECUTE")
    logger.warning(description)
    logger.warning(
        "This command will move physical hardware. You are responsible for ensuring "
        "the workspace is clear, the e-stop is reachable, and the trajectory is safe."
    )
    logger.warning("=" * 70)
    if input("Type 'yes' to proceed: ").strip().lower() != "yes":
        raise SystemExit("Aborted by user.")
    
    
def move_in_force_mode_example(ip: str = "192.168.1.2"):
    """Apply a small +Z compliant force for ~1 s (200 cycles @ 5 ms)."""
    logger.info("Running move_in_force_mode example...")
    robot = universal_robots.UniversalRobotsUR10E()
    robot.connect(ip=ip)
    try:
        task_frame = [0, 0, 0, 0, 0, 0]
        selection_vector = [0, 0, 1, 0, 0, 0]
        wrench = [0, 0, 4, 0, 0, 0]  # 4 N along +Z
        force_type = 2
        limits = [0.4, 0.4, 0.3, 0.2, 0.2, 0.2]
        _confirm_real_motion("Will run force-mode loop for ~1 s with 4 N along +Z")
        for _ in range(200):
            robot.move_in_force_mode(task_frame, selection_vector, wrench, force_type, limits)
            time.sleep(0.005)
        robot.stop_force_mode()
        logger.success("Force mode loop complete.")

    finally:
        robot.disconnect()


def stop_force_mode_example(ip: str = "192.168.1.2"):
    """Stop any active force mode and return to normal position control."""
    logger.info("Running stop_force_mode example...")
    robot = universal_robots.UniversalRobotsUR10E()
    robot.connect(ip=ip)
    try:
        robot.stop_force_mode()
        logger.success("Force mode stopped.")

    finally:
        robot.disconnect()


def set_force_mode_damping_example(ip: str = "192.168.1.2"):
    """Set force-mode damping coefficient to 0.5 (moderate)."""
    logger.info("Running set_force_mode_damping example...")
    robot = universal_robots.UniversalRobotsUR10E()
    robot.connect(ip=ip)
    try:
        robot.set_force_mode_damping(damping=0.5)
        logger.success("Damping set to 0.5.")

    finally:
        robot.disconnect()


def set_force_mode_gain_scaling_example(ip: str = "192.168.1.2"):
    """Set force-mode gain scaling to 0.5 (less responsive, more stable)."""
    logger.info("Running set_force_mode_gain_scaling example...")
    robot = universal_robots.UniversalRobotsUR10E()
    robot.connect(ip=ip)
    try:
        robot.set_force_mode_gain_scaling(scaling=0.5)
        logger.success("Gain scaling set to 0.5.")

    finally:
        robot.disconnect()


def set_external_force_torque_example(ip: str = "192.168.1.2"):
    """Feed an external 10 N along +X into the dynamics model."""
    logger.info("Running set_external_force_torque example...")
    robot = universal_robots.UniversalRobotsUR10E()
    robot.connect(ip=ip)
    try:
        wrench = [10.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        robot.set_external_force_torque(wrench=wrench)
        logger.success(f"External wrench set: {wrench}")

    finally:
        robot.disconnect()


def tool_contact_example(ip: str = "192.168.1.2"):
    """Check whether the tool is contacting in the +Z direction (tool frame)."""
    logger.info("Running tool_contact example...")
    robot = universal_robots.UniversalRobotsUR10E()
    robot.connect(ip=ip)
    try:
        direction = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
        contact = robot.tool_contact(direction=direction)
        logger.success(f"Tool contact: {contact}")

    finally:
        robot.disconnect()


def zero_ft_sensor_example(ip: str = "192.168.1.2"):
    """Zero the F/T sensor — subsequent readings are offset by the current load."""
    logger.info("Running zero_ft_sensor example...")
    robot = universal_robots.UniversalRobotsUR10E()
    robot.connect(ip=ip)
    try:
        robot.zero_ft_sensor()
        logger.success("F/T sensor zeroed.")

    finally:
        robot.disconnect()


def get_example_dict(ip: str = "192.168.1.2"):
    return {
        "move_in_force_mode": lambda: move_in_force_mode_example(ip),
        "stop_force_mode": lambda: stop_force_mode_example(ip),
        "set_force_mode_damping": lambda: set_force_mode_damping_example(ip),
        "set_force_mode_gain_scaling": lambda: set_force_mode_gain_scaling_example(ip),
        "set_external_force_torque": lambda: set_external_force_torque_example(ip),
        "tool_contact": lambda: tool_contact_example(ip),
        "zero_ft_sensor": lambda: zero_ft_sensor_example(ip),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Force Control Synapse examples")
    parser.add_argument("--ip", type=str, default="192.168.1.2")
    parser.add_argument("--example", type=str)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--pause", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    examples = get_example_dict(args.ip)
    if args.list:
        for name in sorted(examples):
            logger.info(f"  - {name}")
        return
    if args.all:
        for name, fn in examples.items():
            logger.info(f"Running {name}...")
            try:
                fn()
            except Exception as e:
                logger.error(f"{name} FAILED: {type(e).__name__}: {e}")
            if args.pause:
                input("Press Enter for next example...")
        return
    if not args.example:
        logger.error("Provide --example, --list, or --all.")
        raise SystemExit(1)
    name = args.example.lower()
    if name not in examples:
        matches = difflib.get_close_matches(name, examples.keys(), n=3, cutoff=0.4)
        logger.error(f"Example '{name}' not found.")
        if matches:
            logger.error("Did you mean: " + ", ".join(matches))
        raise SystemExit(1)
    examples[name]()


if __name__ == "__main__":
    main()
