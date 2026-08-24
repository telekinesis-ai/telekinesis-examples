"""
Move to a target joint configuration along a Cartesian trajectory.

Supports Universal Robots (UR), Epson, and virtual.

Usage:
    python set_joint_position_in_cartesian_space.py
"""

from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots


def main():
    """Move to a target joint configuration along a Cartesian trajectory."""

    #===================== Create Robot ==========================================
    robot = universal_robots.UniversalRobotsUR10E(name='UR10e')

    # ==================== Visualization (Optional) =============================
    robot.visualize_rerun(live=True)

    #===================== Prepare Target ==========================================
    target_joint_positions = robot.get_joint_positions().copy()
    target_joint_positions[0] += 5

    # ==================== Run Skill ============================================
    robot.set_joint_position_in_cartesian_space(
        joint_positions=target_joint_positions,
        speed=1.05,
        acceleration=1.4,
    )
    logger.info(f"Moved to target joint positions: {target_joint_positions}")


if __name__ == "__main__":
    main()
