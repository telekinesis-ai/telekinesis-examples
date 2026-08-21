"""
Attach an OnRobot RG6 gripper to a manipulator and visualize the result.

Supports Universal Robots (UR), Epson, and virtual/sim.

Usage:
    python attach_tool.py
"""

from telekinesis.synapse.robots.manipulators import universal_robots
from telekinesis.synapse.tools.parallel_grippers import onrobot


def main():
    """Attach an OnRobot RG6 gripper to a UR10e and visualize in Rerun."""

    #===================== Create Robot and Gripper =============================
    robot = universal_robots.UniversalRobotsUR10E(name='UR10e')
    gripper = onrobot.OnRobotRG6()

    # ==================== Visualization (Optional) =============================
    robot.visualize_rerun(live=True)

    # ==================== Run Skill ============================================
    # Attach the gripper to the robot and set the active TCP frame
    robot.attach_tool(gripper)
    robot.add_tcp(name="gripper_tip",
                  transform=[0.0, 0.0, 0.18, 0.0, 0.0, 0.0],
                  set_active=True)


if __name__ == "__main__":
    main()
