"""
Example: attach an OnRobot RG6 gripper to a UR10e and visualize in Rerun.

Demonstrates:
    - ``robot.attach_tool()``
    - ``robot.add_tcp()``
    - ``robot.visualize_rerun()``

    Supported for all robots offline, and Universal Robots in real.

Run:
    python examples/synapse/attach_tool/attach_tool.py
"""

from telekinesis.synapse.robots.manipulators import universal_robots
from telekinesis.synapse.tools.parallel_grippers import onrobot



def main():
    """
    Attach an OnRobot RG6 gripper to a UR10e and visualize in Rerun.
    """

    #===================== Create Robot and Gripper =============================
    robot = universal_robots.UniversalRobotsUR10E()
    gripper = onrobot.OnRobotRG6()

    # ==================== Run Skill ============================================
    # Attach the gripper to the robot and set the active TCP frame
    robot.attach_tool(gripper)
    robot.add_tcp(name="gripper_tip",
                  transform=[0.0, 0.0, 0.18, 0.0, 0.0, 0.0],
                  set_active=True)

    # ==================== Visualization (Optional) =============================
    robot.visualize_rerun()


if __name__ == "__main__":
    main()
