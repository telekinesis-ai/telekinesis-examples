"""
Solve inverse kinematics for a fixed target TCP pose with a default seed.

Supports Universal Robots (UR), Epson, and virtual.

Usage:
    python inverse_kinematics.py
"""

from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots


def main():
    """Solve IK for a target TCP pose with default solver parameters."""

    #===================== Create Robot ==========================================
    robot = universal_robots.UniversalRobotsUR10E(name='UR10e')

    # ==================== Run Skill ============================================
    # Solve IK for a fixed target pose [x, y, z, rx, ry, rz] (m, deg)
    target_pose = [0.3, 0.3, 0.3, 180, 0, 0]
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
