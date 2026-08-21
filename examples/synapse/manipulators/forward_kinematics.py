"""
Compute forward kinematics for a fixed joint configuration.

Supports Universal Robots (UR), Epson, and virtual/sim.

Usage:
    python forward_kinematics.py
"""

from telekinesis.synapse.robots.manipulators import universal_robots


def main():
    """Compute forward kinematics for a fixed joint configuration and visualize the result."""

    #===================== Create Robot ==========================================
    robot = universal_robots.UniversalRobotsUR10E(name='UR10e')

    # ==================== Run Skill ============================================
    q = [0, -90, 90, 0, 90, 0]
    tcp_pose = robot.forward_kinematics(q=q)
    print("TCP pose: ", tcp_pose)

    # ==================== Visualization (Optional) =============================
    robot.set_joint_positions(joint_positions=q)
    robot.visualize_rerun(live=False)


if __name__ == "__main__":
    main()
