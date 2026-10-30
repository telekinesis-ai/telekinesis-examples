"""
Change the stored settings of the active IK solver, then solve IK with them.

Supports Universal Robots (UR), Epson, and virtual.

Usage:
    python set_kinematics_settings.py
"""

from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots


def main():
    """Tighten the CLIK settings and solve IK with the updated solver."""

    #===================== Create Robot ==========================================
    robot = universal_robots.UniversalRobotsUR10E(name='UR10e')

    # An IK solver must be active before its settings can be changed
    robot.setup_kinematics_solver(solver="clik")
    logger.info(f"Settings before: {robot.get_kinematics_settings()}")

    # ==================== Run Skill ============================================
    # Omitted settings keep their stored values
    robot.set_kinematics_settings(timeout_s=0.1, eps=1e-5, it_max=2000)
    logger.info(f"Settings after: {robot.get_kinematics_settings()}")

    # Later IK calls use the updated settings without passing them again
    target_pose = [0.5, 0.2, 0.3, 180.0, 0.0, 0.0]
    try:
        q = robot.inverse_kinematics(target_poses=[target_pose])[0]
        if q is None:
            logger.error(f"No IK solution for {target_pose}")
            return
        logger.success(f"IK solution: {q}")

        # ================ Visualization (Optional) ==============================
        robot.set_joint_positions(joint_positions=q)
        robot.visualize_rerun(live=False)
    except (RuntimeError, TypeError, ValueError) as e:
        logger.error(f"IK failed: {type(e).__name__}: {e}")
    finally:
        robot.shutdown()


if __name__ == "__main__":
    main()
