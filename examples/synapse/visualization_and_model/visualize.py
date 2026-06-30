"""
Visualize a robot in rerun

Run:
    python examples/synapse/visualization_and_model/visualize.py
"""
import time

import numpy as np
import rerun as rr
from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots


def main():
    """Visualize robot in Rerun"""

    # Initialize telekinesis-synapse UR10e robot
    robot = universal_robots.UniversalRobotsUR10E()

    # Visualize robot
    robot.visualize_rerun()

if __name__ == "__main__":
    main()
