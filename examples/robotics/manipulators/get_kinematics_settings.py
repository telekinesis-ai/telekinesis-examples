"""
Read the stored settings of the active IK solver.

Supports Universal Robots (UR), Epson, and virtual.

Usage:
    python get_kinematics_settings.py
"""

from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots


def main():
    """Select an IK solver and log its stored settings."""

    #===================== Create Robot ==========================================
    robot = universal_robots.UniversalRobotsUR10E(name='UR10e')

    # An IK solver must be active before its settings can be read
    robot.setup_kinematics_solver(solver="clik")

    # ==================== Run Skill ============================================
    settings = robot.get_kinematics_settings()
    logger.info(f"Active IK solver: {settings.solver_name}")
    logger.info(f"Time budget: {settings.timeout_s} s")
    logger.info(f"Convergence tolerance: {settings.eps}")

    if settings.solver_name == "tracik":
        logger.info(f"TRAC-IK search mode: {settings.tracik_solve_type}")
    else:
        logger.info(f"Iterations per CLIK attempt: {settings.it_max}")
        logger.info(f"CLIK integration step: {settings.dt}")
        logger.info(f"CLIK damping: {settings.damp}")

    robot.shutdown()


if __name__ == "__main__":
    main()
