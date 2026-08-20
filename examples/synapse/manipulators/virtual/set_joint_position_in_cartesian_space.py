"""
Set Joint Position in Cartesian space example for the Synapse SDK -- offline.

Moves to a target configuration using Cartesian motion derived from joint
positions (as opposed to ``set_joint_positions``, which moves in joint
space). Offline, the configuration is resolved to a TCP pose and played back
through the control loop on the kinematic model; no hardware connection is
made.

Supports all robots.

Usage:
    python set_joint_position_in_cartesian_space.py
"""

from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots


def main():
    """Move to a target joint configuration along a Cartesian trajectory."""

    #===================== Create Robot ==========================================
    robot = universal_robots.UniversalRobotsUR10E()

    # ==================== Run Skill ============================================
    # Target: current joint configuration with the base joint rotated
    target_joint_positions = robot.get_joint_positions().copy()
    target_joint_positions[0] += 5

    robot.set_joint_position_in_cartesian_space(
        joint_positions=target_joint_positions,
        speed=1.05,
        acceleration=1.4,
    )
    logger.info(f"Moved to target joint positions: {target_joint_positions}")

    # ==================== Visualization (Optional) =============================
    robot.visualize_rerun(live=False)


if __name__ == "__main__":
    main()
