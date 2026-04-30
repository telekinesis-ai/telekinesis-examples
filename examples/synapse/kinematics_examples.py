"""Kinematics examples for the Synapse SDK on Universal Robots UR10e.

Each example is self-contained: it constructs its own robot, connects, runs the
skill, and disconnects. Run a single example with --example <name>, list all
with --list, or run them all with --all.
"""

import argparse
import difflib

from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots


def forward_kinematics_example(ip: str = "192.168.1.2"):
    """Compute forward kinematics for a given joint configuration (offline; no robot needed)."""
    logger.info("Running forward_kinematics example...")
    robot = universal_robots.UniversalRobotsUR10E()
    q = [0, -90, 90, 0, 90, 0]
    tcp_pose = robot.forward_kinematics(q)
    logger.success(f"FK TCP pose for q={q}: {tcp_pose}")


def inverse_kinematics_example(ip: str = "192.168.1.2"):
    """Compute inverse kinematics for a target Cartesian pose (offline; no robot needed)."""
    logger.info("Running inverse_kinematics example...")
    robot = universal_robots.UniversalRobotsUR10E()
    target_pose = [0.3, 0.3, 0.3, 180, 0, 0]
    try:
        q = robot.inverse_kinematics(target_pose=target_pose, solver='multi_start_clik')
        logger.success(f"IK solution for {target_pose}: {q}")
    except (RuntimeError, TypeError, ValueError) as e:
        logger.error(f"IK failed: {e}")


def get_on_robot_forward_kinematics_example(ip: str = "192.168.1.2"):
    """Compute forward kinematics on the robot controller."""
    logger.info("Running get_on_robot_forward_kinematics example...")
    robot = universal_robots.UniversalRobotsUR10E()
    robot.connect(ip=ip)
    try:
        q = [0.0, -90.0, -90.0, 0.0, 90.0, 0.0]
        result = robot.get_on_robot_forward_kinematics(q, tcp_offset=None)
        logger.success(f"On-robot FK for q={q}: {result}")
    finally:
        robot.disconnect()


def get_on_robot_inverse_kinematics_example(ip: str = "192.168.1.2"):
    """Compute inverse kinematics on the robot controller."""
    logger.info("Running get_on_robot_inverse_kinematics example...")
    robot = universal_robots.UniversalRobotsUR10E()
    robot.connect(ip=ip)
    try:
        pose = [0.5, 0.0, 0.5, 0.0, 0.0, 0.0]
        result = robot.get_on_robot_inverse_kinematics(pose, qnear=None)
        logger.success(f"On-robot IK for pose={pose}: {result}")
    finally:
        robot.disconnect()


def get_inverse_kinematics_has_solution_example(ip: str = "192.168.1.2"):
    """Check whether an IK solution exists for the current Cartesian pose."""
    logger.info("Running get_inverse_kinematics_has_solution example...")
    robot = universal_robots.UniversalRobotsUR10E()
    robot.connect(ip=ip)
    try:
        current_pose = robot.get_cartesian_pose()
        has_solution = robot.get_inverse_kinematics_has_solution(current_pose)
        logger.success(f"IK solution exists for current pose: {has_solution}")
    finally:
        robot.disconnect()


def get_jacobian_example(ip: str = "192.168.1.2"):
    """Read the current 6x6 Jacobian (returned as a 36-element flat list)."""
    logger.info("Running get_jacobian example...")
    robot = universal_robots.UniversalRobotsUR10E()
    robot.connect(ip=ip)
    try:
        jacobian = robot.get_jacobian()
        logger.success(f"Jacobian (6x6 flat): {jacobian}")
    finally:
        robot.disconnect()


def get_jacobian_time_derivative_example(ip: str = "192.168.1.2"):
    """Read the current Jacobian time derivative."""
    logger.info("Running get_jacobian_time_derivative example...")
    robot = universal_robots.UniversalRobotsUR10E()
    robot.connect(ip=ip)
    try:
        jdot = robot.get_jacobian_time_derivative()
        logger.success(f"Jacobian time derivative: {jdot}")
    finally:
        robot.disconnect()


def get_actual_tool_flange_pose_example(ip: str = "192.168.1.2"):
    """Read the tool flange pose (excluding any TCP offset)."""
    logger.info("Running get_actual_tool_flange_pose example...")
    robot = universal_robots.UniversalRobotsUR10E()
    robot.connect(ip=ip)
    try:
        flange_pose = robot.get_actual_tool_flange_pose()
        logger.success(f"Tool flange pose [x, y, z, rx, ry, rz]: {flange_pose}")
    finally:
        robot.disconnect()


def get_example_dict(ip: str = "192.168.1.2"):
    """Mapping of example name -> zero-arg callable that runs the example."""
    return {
        "forward_kinematics": lambda: forward_kinematics_example(ip),
        "inverse_kinematics": lambda: inverse_kinematics_example(ip),
        "get_on_robot_forward_kinematics": lambda: get_on_robot_forward_kinematics_example(ip),
        "get_on_robot_inverse_kinematics": lambda: get_on_robot_inverse_kinematics_example(ip),
        "get_inverse_kinematics_has_solution": lambda: get_inverse_kinematics_has_solution_example(ip),
        "get_jacobian": lambda: get_jacobian_example(ip),
        "get_jacobian_time_derivative": lambda: get_jacobian_time_derivative_example(ip),
        "get_actual_tool_flange_pose": lambda: get_actual_tool_flange_pose_example(ip),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Kinematics Synapse examples")
    parser.add_argument("--ip", type=str, default="192.168.1.2", help="UR robot IP")
    parser.add_argument("--example", type=str, help="Example name to run (or use --list)")
    parser.add_argument("--list", action="store_true", help="List all examples")
    parser.add_argument("--all", action="store_true", help="Run all examples")
    parser.add_argument("--pause", action="store_true", help="Pause between examples when --all")
    return parser.parse_args()


def main():
    args = parse_args()
    examples = get_example_dict(args.ip)
    if args.list:
        logger.info("Available examples:")
        for name in sorted(examples):
            logger.info(f"  - {name}")
        return
    if args.all:
        for name, fn in examples.items():
            logger.info(f"Running {name}...")
            try:
                fn()
                logger.success(f"{name} done.")
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
    logger.info(f"Running {name}...")
    examples[name]()
    logger.success(f"{name} done.")


if __name__ == "__main__":
    main()
