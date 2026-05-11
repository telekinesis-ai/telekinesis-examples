"""
Manipulator state examples for the Synapse SDK.

Connect to a robot with --ip to read live hardware state- Currently supported
only for Universal Robots (UR10e). 

These examples read the brand-agnostic ManipulatorState through the
robot's ``get_*`` API when not connected.

Usage for real hardware:
    python get_manipulator_states.py --list
    python get_manipulator_states.py [--ip <ROBOT_IP>] --example <NAME>
    python get_manipulator_states.py [--ip <ROBOT_IP>] --all

Usage without hardware (reads from internal commanded cache, logs a warning):
    python get_manipulator_states.py --list
    python get_manipulator_states.py --example <NAME>
    python get_manipulator_states.py --all

Use --list to print the names of available examples without connecting to a
robot. When --ip is omitted, the manipulator is instantiated but never
connected, so every read comes from the internal commanded cache and a
warning is logged.
"""

import argparse
import difflib

from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots


def is_connected_example(robot):
    """Read whether the manipulator state is live (hardware-driven)."""
    logger.success(f"is_connected: {robot.is_connected()}")


def joint_positions_example(robot):
    """Read joint positions [deg]."""
    logger.success(f"joint_positions [deg]: {robot.get_joint_positions()}")


def joint_velocities_example(robot):
    """Read joint velocities [deg/s]."""
    logger.success(f"joint_velocities [deg/s]: {robot.get_actual_joint_velocities()}")


def joint_torques_example(robot):
    """Read joint torques [N·m]."""
    logger.success(f"joint_torques [N·m]: {robot.get_joint_torques()}")


def tcp_pose_example(robot):
    """Read TCP pose [x, y, z (m), rx, ry, rz (deg)]."""
    logger.success(f"tcp_pose [m, deg]: {robot.get_cartesian_pose()}")


def tcp_speed_example(robot):
    """Read TCP velocity [vx, vy, vz (m/s), ωx, ωy, ωz (deg/s)]."""
    logger.success(f"tcp_speed [m/s, deg/s]: {robot.get_actual_tcp_speed()}")


def tcp_force_example(robot):
    """Read TCP wrench [Fx, Fy, Fz (N), Tx, Ty, Tz (N·m)]."""
    logger.success(f"tcp_force [N, N·m]: {robot.get_actual_tcp_force()}")


def target_joint_positions_example(robot):
    """Read target (commanded) joint positions [deg]."""
    logger.success(f"target_joint_positions [deg]: {robot.get_target_joint_positions()}")


def target_joint_velocities_example(robot):
    """Read target (commanded) joint velocities [deg/s]."""
    logger.success(f"target_joint_velocities [deg/s]: {robot.get_target_joint_velocities()}")


def target_joint_accelerations_example(robot):
    """Read target (commanded) joint accelerations [deg/s²]."""
    logger.success(f"target_joint_accelerations [deg/s²]: {robot.get_target_joint_accelerations()}")


def target_tcp_pose_example(robot):
    """Read target (commanded) TCP pose [x, y, z (m), rx, ry, rz (deg)]."""
    logger.success(f"target_tcp_pose [m, deg]: {robot.get_target_tcp_pose()}")


def target_tcp_speed_example(robot):
    """Read target (commanded) TCP velocity [vx, vy, vz (m/s), ωx, ωy, ωz (deg/s)]."""
    logger.success(f"target_tcp_speed [m/s, deg/s]: {robot.get_target_tcp_speed()}")


def timestamp_example(robot):
    """Read the timestamp of the most recent state update [s since epoch]."""
    logger.success(f"timestamp [s]: {robot.get_timestamp()}")


def get_example_dict(robot):
    return {
        "is_connected": lambda: is_connected_example(robot),
        "joint_positions": lambda: joint_positions_example(robot),
        "joint_velocities": lambda: joint_velocities_example(robot),
        "joint_torques": lambda: joint_torques_example(robot),
        "tcp_pose": lambda: tcp_pose_example(robot),
        "tcp_speed": lambda: tcp_speed_example(robot),
        "tcp_force": lambda: tcp_force_example(robot),
        "target_joint_positions": lambda: target_joint_positions_example(robot),
        "target_joint_velocities": lambda: target_joint_velocities_example(robot),
        "target_joint_accelerations": lambda: target_joint_accelerations_example(robot),
        "target_tcp_pose": lambda: target_tcp_pose_example(robot),
        "target_tcp_speed": lambda: target_tcp_speed_example(robot),
        "timestamp": lambda: timestamp_example(robot),
    }


def main():
    """
    Run a Manipulator state Synapse example.
    Usage:
        python get_manipulator_states.py --list
        python get_manipulator_states.py [--ip <ROBOT_IP>] --example <NAME>
        python get_manipulator_states.py [--ip <ROBOT_IP>] --all

    --ip is fully optional. With it, the robot is connected for the
    duration of the example(s). Without it, the robot is instantiated but
    never connected, so reads come from the internal commanded cache and a
    warning is logged.
    """

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Manipulator state Synapse examples")
    parser.add_argument(
        "--ip",
        type=str,
        help="UR robot IP address (omit to read offline commanded-cache state)",
    )
    parser.add_argument("--example", type=str)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    # --list is a name-only operation that doesn't need a robot instance
    if args.list:
        for name in sorted(get_example_dict(robot=None)):
            logger.info(f"  - {name}")
        return

    # Create the robot and connect if --ip was provided
    robot = universal_robots.UniversalRobotsUR10E()
    if args.ip:
        robot.connect(ip=args.ip)
    else:
        logger.warning(
            "No --ip provided; reading offline commanded-cache state, "
            "not live hardware readings."
        )

    # Run example(s), then disconnect if we connected
    try:
        examples = get_example_dict(robot)
        if args.all:
            for name, fn in examples.items():
                logger.info(f"Running {name}...")
                try:
                    fn()
                except Exception as e:
                    logger.error(f"{name} FAILED: {type(e).__name__}: {e}")
            return

        # Handle single example execution
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

    finally:
        if args.ip:
            robot.disconnect()


if __name__ == "__main__":
    main()
