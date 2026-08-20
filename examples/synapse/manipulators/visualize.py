"""
Visualize a robot in Rerun.

Run:
    python examples/synapse/visualization_and_model/visualize.py
"""
from telekinesis.synapse.robots.manipulators import universal_robots


def main():
    """Visualize robot in Rerun"""

    #===================== Create Robot ==========================================
    robot = universal_robots.UniversalRobotsUR10E()

    # ==================== Run Skill ============================================
    robot.visualize_rerun()

if __name__ == "__main__":
    main()
