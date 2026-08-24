"""
Pre-initialize an IK solver, then solve IK with it.

Supports Universal Robots (UR), Epson, and virtual.

Usage:
    python setup_kinematics_solver.py
"""

from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots


def main():
    """Pre-initialize the multi_start_clik solver, then solve IK with it."""

    #===================== Create Robot ==========================================
    robot = universal_robots.UniversalRobotsUR10E(name='UR10e')

    # ==================== Run Skill ============================================
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

        # ================ Visualization (Optional) ==============================
        robot.set_joint_positions(joint_positions=q)
        robot.visualize_rerun(live=False)
    except (RuntimeError, TypeError, ValueError) as e:
        logger.error(f"IK failed: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
