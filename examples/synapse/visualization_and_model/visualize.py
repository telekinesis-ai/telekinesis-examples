"""
Visualize a robot in Rerun.

Run:
    python examples/synapse/visualization_and_model/visualize.py
"""
from telekinesis.synapse.robots.manipulators import universal_robots


def main():
    """Visualize robot in Rerun"""

    # Initialize telekinesis-synapse UR10e robot
    robot = universal_robots.UniversalRobotsUR10E()

    # Visualize robot
    robot.visualize_rerun()

if __name__ == "__main__":
    main()
