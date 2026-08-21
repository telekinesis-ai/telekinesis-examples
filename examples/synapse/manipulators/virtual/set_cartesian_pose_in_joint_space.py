"""
Move the TCP to a target pose along a joint-space-linear trajectory.

Supports Universal Robots (UR), Epson, and virtual/sim.

Usage:
    python set_cartesian_pose_in_joint_space.py
"""

from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots


def main():
    """Move the TCP to a target pose along a joint-space-linear trajectory."""

    #===================== Create Robot ==========================================
    robot = universal_robots.UniversalRobotsUR10E(name='UR10e')

    # ==================== Visualization (Optional) =============================
    robot.visualize_rerun(live=True)

    #===================== Prepare Target ==========================================
    current_cartesian_pose = robot.get_cartesian_pose()
    target_cartesian_pose = current_cartesian_pose.copy()
    target_cartesian_pose[2] += 0.1  # Move 10 cm up in Z

    # ==================== Run Skill ============================================
    robot.set_cartesian_pose_in_joint_space(
        cartesian_pose=target_cartesian_pose,
        speed=60,
        acceleration=80,
    )
    logger.info(f"Moved to target Cartesian pose: {target_cartesian_pose}")


if __name__ == "__main__":
    main()
