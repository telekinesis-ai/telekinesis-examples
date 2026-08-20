"""
Set Cartesian Pose in joint space example for the Synapse SDK -- offline.

Moves the TCP to a target Cartesian pose with a trajectory linear in joint
space (as opposed to ``set_cartesian_pose``, which is linear in Cartesian
space). Offline, the pose is solved to a joint configuration and played back
through the control loop on the kinematic model; no hardware connection is
made.

Supports all robots.

Usage:
    python set_cartesian_pose_in_joint_space.py
"""

from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots


def main():
    """Move the TCP to a target pose along a joint-space-linear trajectory."""

    #===================== Create Robot ==========================================
    robot = universal_robots.UniversalRobotsUR10E()

    # ==================== Run Skill ============================================
    # Define a target relative to the current pose
    current_cartesian_pose = robot.get_cartesian_pose()
    target_cartesian_pose = current_cartesian_pose.copy()
    target_cartesian_pose[2] += 0.1  # Move 10 cm up in Z

    robot.set_cartesian_pose_in_joint_space(
        cartesian_pose=target_cartesian_pose,
        speed=60,
        acceleration=80,
    )
    logger.info(f"Moved to target Cartesian pose: {target_cartesian_pose}")

    # ==================== Visualization (Optional) =============================
    robot.visualize_rerun(live=False)


if __name__ == "__main__":
    main()
