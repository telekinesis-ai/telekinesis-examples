"""Motion examples for the Synapse SDK on Universal Robots UR10e."""

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

def set_cartesian_pose_example(ip: str = "192.168.1.2"):
    """Move TCP to a small Cartesian-space delta from current pose."""
    logger.info("Running set_cartesian_pose example...")
    robot = universal_robots.UniversalRobotsUR10E()
    robot.connect(ip=ip)
    try:
        current = robot.get_cartesian_pose()
        target = list(current); target[2] += 0.02  # +2cm Z
        _confirm_real_motion(f"Will set Cartesian pose to {target} (current {current})")
        robot.set_cartesian_pose(cartesian_pose=target, speed=0.21, acceleration=0.28)
        logger.success(f"Moved to {target}")

    finally:
        robot.disconnect()


def set_joint_positions_example(ip: str = "192.168.1.2"):
    """Move joint 5 by +5 deg from current configuration."""
    logger.info("Running set_joint_positions example...")
    robot = universal_robots.UniversalRobotsUR10E()
    robot.connect(ip=ip)
    try:
        q = list(robot.get_joint_positions())
        q[4] += 5.0
        _confirm_real_motion(f"Will set joint positions to {q}")
        robot.set_joint_positions(joint_positions=q, speed=12, acceleration=16)
        logger.success(f"Moved to {q}")

    finally:
        robot.disconnect()


def set_cartesian_pose_in_joint_space_example(ip: str = "192.168.1.2"):
    """Move to a small Cartesian-pose delta but plan in joint space."""
    logger.info("Running set_cartesian_pose_in_joint_space example...")
    robot = universal_robots.UniversalRobotsUR10E()
    robot.connect(ip=ip)
    try:
        current = robot.get_cartesian_pose()
        target = list(current); target[2] += 0.02
        _confirm_real_motion(f"Will set Cartesian (joint-space) to {target}")
        robot.set_cartesian_pose_in_joint_space(cartesian_pose=target, speed=12, acceleration=16)
        logger.success(f"Moved to {target}")

    finally:
        robot.disconnect()


def set_joint_position_in_cartesian_space_example(ip: str = "192.168.1.2"):
    """Move to small joint-position delta but plan/execute in Cartesian space."""
    logger.info("Running set_joint_position_in_cartesian_space example...")
    robot = universal_robots.UniversalRobotsUR10E()
    robot.connect(ip=ip)
    try:
        q = list(robot.get_joint_positions())
        q[4] += 5.0
        _confirm_real_motion(f"Will set joint-position-in-cartesian to {q}")
        robot.set_joint_position_in_cartesian_space(joint_positions=q, speed=0.21, acceleration=0.28)
        logger.success(f"Moved to {q}")

    finally:
        robot.disconnect()


def move_joint_path_example(ip: str = "192.168.1.2"):
    """Step through a 3-waypoint joint path with small deltas from current q."""
    logger.info("Running move_joint_path example...")
    robot = universal_robots.UniversalRobotsUR10E()
    robot.connect(ip=ip)
    try:
        q0 = list(robot.get_joint_positions())
        path = [list(q0), [q0[0]+5, *q0[1:]], [q0[0]+10, *q0[1:]]]
        _confirm_real_motion(f"Will move through {len(path)} joint waypoints")
        robot.move_joint_path(path=path)
        logger.success("Joint path complete")

    finally:
        robot.disconnect()


def move_cartesian_path_example(ip: str = "192.168.1.2"):
    """Move TCP through a 3-waypoint path offset along Z."""
    logger.info("Running move_cartesian_path example...")
    robot = universal_robots.UniversalRobotsUR10E()
    robot.connect(ip=ip)
    try:
        current = robot.get_cartesian_pose()
        path = [
            list(current),
            [current[0], current[1], current[2]+0.01, *current[3:]],
            [current[0], current[1], current[2]+0.02, *current[3:]],
        ]
        _confirm_real_motion(f"Will move through {len(path)} Cartesian waypoints")
        robot.move_cartesian_path(path=path)
        logger.success("Cartesian path complete")

    finally:
        robot.disconnect()


def move_path_example(ip: str = "192.168.1.2"):
    """Execute a pre-built rtde_control.Path from current joint configuration."""
    logger.info("Running move_path example...")
    robot = universal_robots.UniversalRobotsUR10E()
    robot.connect(ip=ip)
    try:
        import math, rtde_control
        q0 = robot.get_joint_positions()
        path = rtde_control.Path()
        for delta in [0.0, 5.0]:
            q_target = [math.radians(q0[0]+delta)] + [math.radians(v) for v in q0[1:]] + [0.21, 0.28, 0.0]
            path.addEntry(rtde_control.PathEntry(rtde_control.PathEntry.eMoveJ, rtde_control.PathEntry.ePositionJoints, q_target))
        _confirm_real_motion("Will execute pre-built Path")
        robot.move_path(path=path)
        logger.success("Path complete")

    finally:
        robot.disconnect()


def move_until_contact_example(ip: str = "192.168.1.2"):
    """Move slowly downward (-Z, 1 cm/s) until contact is detected."""
    logger.info("Running move_until_contact example...")
    robot = universal_robots.UniversalRobotsUR10E()
    robot.connect(ip=ip)
    try:
        _confirm_real_motion("Will move TCP downward at 1 cm/s until contact")
        contacted = robot.move_until_contact(
            cartesian_speed=[0, 0, -0.01, 0, 0, 0],
            direction=[0, 0, 0, 0, 0, 0],
            acceleration=0.05,
        )
        logger.success(f"Contact: {contacted}")

    finally:
        robot.disconnect()


def stop_cartesian_motion_example(ip: str = "192.168.1.2"):
    """Stop any active Cartesian motion (safe to call when idle)."""
    logger.info("Running stop_cartesian_motion example...")
    robot = universal_robots.UniversalRobotsUR10E()
    robot.connect(ip=ip)
    try:
        robot.stop_cartesian_motion(stopping_speed=0.5)
        logger.success("Cartesian motion stopped.")

    finally:
        robot.disconnect()


def stop_joint_motion_example(ip: str = "192.168.1.2"):
    """Stop any active joint motion (safe to call when idle)."""
    logger.info("Running stop_joint_motion example...")
    robot = universal_robots.UniversalRobotsUR10E()
    robot.connect(ip=ip)
    try:
        robot.stop_joint_motion(stopping_speed=0.5)
        logger.success("Joint motion stopped.")

    finally:
        robot.disconnect()


def trigger_protective_stop_example(ip: str = "192.168.1.2"):
    """Trigger a protective stop on the robot (requires unlock_protective_stop afterward)."""
    logger.info("Running trigger_protective_stop example...")
    robot = universal_robots.UniversalRobotsUR10E()
    robot.connect(ip=ip)
    try:
        _confirm_real_motion("Will TRIGGER PROTECTIVE STOP — robot will halt")
        robot.trigger_protective_stop()
        logger.success("Protective stop triggered.")

    finally:
        robot.disconnect()


def start_jog_example(ip: str = "192.168.1.2"):
    """Start jogging joint 1 at 1 deg/s for 1 s, then stop."""
    logger.info("Running start_jog example...")
    robot = universal_robots.UniversalRobotsUR10E()
    robot.connect(ip=ip)
    try:
        speeds = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        _confirm_real_motion(f"Will start jog at {speeds} deg/s for 1 second")
        robot.start_jog(speeds=speeds, feature=0)
        time.sleep(1.0)
        robot.stop_jog()
        logger.success("Jog stopped.")

    finally:
        robot.disconnect()


def stop_jog_example(ip: str = "192.168.1.2"):
    """Stop any active jog motion."""
    logger.info("Running stop_jog example...")
    robot = universal_robots.UniversalRobotsUR10E()
    robot.connect(ip=ip)
    try:
        robot.stop_jog()
        logger.success("Jog stopped.")

    finally:
        robot.disconnect()


def start_freedrive_mode_example(ip: str = "192.168.1.2"):
    """Enter freedrive (back-drivable) mode for 5 seconds, then exit."""
    logger.info("Running start_freedrive_mode example...")
    robot = universal_robots.UniversalRobotsUR10E()
    robot.connect(ip=ip)
    try:
        _confirm_real_motion("Will enter FREEDRIVE for 5 seconds — robot becomes back-drivable")
        robot.start_freedrive_mode(free_axes=[1, 1, 1, 1, 1, 1])
        time.sleep(5.0)
        robot.stop_freedrive_mode()
        logger.success("Freedrive stopped.")

    finally:
        robot.disconnect()


def stop_freedrive_mode_example(ip: str = "192.168.1.2"):
    """Exit freedrive mode and return to normal control."""
    logger.info("Running stop_freedrive_mode example...")
    robot = universal_robots.UniversalRobotsUR10E()
    robot.connect(ip=ip)
    try:
        robot.stop_freedrive_mode()
        logger.success("Freedrive stopped.")

    finally:
        robot.disconnect()


def start_teach_mode_example(ip: str = "192.168.1.2"):
    """Enter teach mode for 5 seconds, then exit."""
    logger.info("Running start_teach_mode example...")
    robot = universal_robots.UniversalRobotsUR10E()
    robot.connect(ip=ip)
    try:
        _confirm_real_motion("Will enter TEACH MODE for 5 seconds — robot becomes back-drivable")
        robot.start_teach_mode()
        time.sleep(5.0)
        robot.stop_teach_mode()
        logger.success("Teach mode stopped.")

    finally:
        robot.disconnect()


def stop_teach_mode_example(ip: str = "192.168.1.2"):
    """Exit teach mode and return to normal control."""
    logger.info("Running stop_teach_mode example...")
    robot = universal_robots.UniversalRobotsUR10E()
    robot.connect(ip=ip)
    try:
        robot.stop_teach_mode()
        logger.success("Teach mode stopped.")

    finally:
        robot.disconnect()


def get_example_dict(ip: str = "192.168.1.2"):
    return {
        "set_cartesian_pose": lambda: set_cartesian_pose_example(ip),
        "set_joint_positions": lambda: set_joint_positions_example(ip),
        "set_cartesian_pose_in_joint_space": lambda: set_cartesian_pose_in_joint_space_example(ip),
        "set_joint_position_in_cartesian_space": lambda: set_joint_position_in_cartesian_space_example(ip),
        "move_joint_path": lambda: move_joint_path_example(ip),
        "move_cartesian_path": lambda: move_cartesian_path_example(ip),
        "move_path": lambda: move_path_example(ip),
        "move_until_contact": lambda: move_until_contact_example(ip),
        "stop_cartesian_motion": lambda: stop_cartesian_motion_example(ip),
        "stop_joint_motion": lambda: stop_joint_motion_example(ip),
        "trigger_protective_stop": lambda: trigger_protective_stop_example(ip),
        "start_jog": lambda: start_jog_example(ip),
        "stop_jog": lambda: stop_jog_example(ip),
        "start_freedrive_mode": lambda: start_freedrive_mode_example(ip),
        "stop_freedrive_mode": lambda: stop_freedrive_mode_example(ip),
        "start_teach_mode": lambda: start_teach_mode_example(ip),
        "stop_teach_mode": lambda: stop_teach_mode_example(ip),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Motion Synapse examples")
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
