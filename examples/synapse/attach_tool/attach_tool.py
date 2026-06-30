"""
Example: Attach tool and visualize in Rerun

Run:
    python examples/synapse/attach_tool/attach_tool.py
"""

from telekinesis.synapse.robots.manipulators import universal_robots
from telekinesis.synapse.tools.parallel_grippers import onrobot



def main():
    """
    Attach robotiq gripper to UniversalRobots UR10e and visualize in Rerun.
    """
  
    robot = universal_robots.UniversalRobotsUR10E()
    gripper = onrobot.OnRobotRG6()
   
    robot.attach_tool(gripper)
    robot.add_tcp(name="gripper_tip",
                  transform=[0.0, 0.0, 0.18, 0.0, 0.0, 0.0],
                  set_active=True)

    # Visualize
    robot.visualize_rerun()


if __name__ == "__main__":
    main()
