"""
Visualize a robot in Rerun.

Supports Universal Robots (UR), Epson, and virtual.

Usage:
    python visualize.py
"""
from telekinesis.synapse.robots.manipulators import universal_robots


def main():
    """Visualize the robot live in Rerun."""

    #===================== Create Robot ==========================================
    robot = universal_robots.UniversalRobotsUR10E(name='UR10e')

    # ==================== Run Skill ============================================
    robot.visualize_rerun(live=True)


if __name__ == "__main__":
    main()
