"""
Forward kinematics example for the Synapse SDK.

Compute the forward kinematics for a manipulator using the Synapse SDK.

Universal Robots (UR10e) is used here purely for illustration; the same API works for all supported robots.

Usage:
    python forward_kinematics.py
"""

from telekinesis.synapse.robots.manipulators import universal_robots


def main():
    """
    Demonstrates forward kinematics computation for the Universal Robot UR10e
    """

    #===================== Create Robot ==========================================
    robot = universal_robots.UniversalRobotsUR10E()

    # ==================== Run Skill ============================================
    q = [0, -90, 90, 0, 90, 0]
    tcp_pose = robot.forward_kinematics(q=q)
    print("TCP pose: ", tcp_pose)

    # ==================== Visualization (Optional) =============================
    robot.set_joint_positions(joint_positions=q)
    robot.visualize_rerun(live=False)


if __name__ == "__main__":
    main()
