"""
Pre-initialize an IK solver for the Synapse SDK.

``setup_kinematics_solver`` initializes an IK solver by name and caches
it on the robot, so subsequent ``inverse_kinematics`` calls skip the
solver-construction cost. 

Supported solver names: ``"clik"``, ``"multi_start_clik"``, ``"tracik"``.

Universal Robots (UR10e) is used here purely for illustration. It supports all robots.

Usage:
    python setup_kinematics_solver.py
"""

from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots


def main():
    """Pre-initialize the multi_start_clik solver, then solve IK with it."""

    # Create the robot (no connect required — IK runs on the kinematic model)
    robot = universal_robots.UniversalRobotsUR10E()

    # Get all supported kinematics solvers
    solvers = robot.supported_kinematics_solvers
    logger.info(f"Supported solvers: {solvers}")

    # Pre-load the desired solver so the first inverse_kinematics call is fast
    robot.setup_kinematics_solver(solver="multi_start_clik")

    # Check active kinematic solver
    active_kinematics_solver = robot.active_kinematics_solver
    logger.info(f"Active kinematics solver: {active_kinematics_solver}")

    # Solve IK using the cached solver (no need to pass ``solver=`` again)
    target_pose = [0.5, 0.2, 0.3, 180.0, 0.0, 0.0]
    try:
        q = robot.inverse_kinematics(target_pose=target_pose)
        logger.success(f"IK solution: {q}")
    except (RuntimeError, TypeError, ValueError) as e:
        logger.error(f"IK failed: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
