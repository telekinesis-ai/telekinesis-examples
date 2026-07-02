"""
Example: visualize a UR10e robot with multiple tools.

Demonstrates:
    - ``robot.attach_tool(gripper)``     -- attach gripper once; visualization is automatic
    - ``robot.visualize_rerun()``        -- renders robot + gripper together every step

    Supported for all robots offline, and Universal Robots in real.

Run:
    python examples/synapse/attach_tool/attach_and_visualize_all_tools_supported.py

"""

import time
from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots
from telekinesis.synapse.tools.parallel_grippers import robotiq, onrobot, schunk

_GRIPPERS = {
    "2f85":  robotiq.Robotiq2F85,  
    "2f140": robotiq.Robotiq2F140, 
    "hande": robotiq.RobotiqHandE, 
    "rg6":   onrobot.OnRobotRG6,
    "rg2":   onrobot.OnRobotRG2,
    "egu50":   schunk.SchunkEGU50,
    "egp":     schunk.SchunkEGP,
    "pznplus": schunk.SchunkPZNPlus,
    "pzv64":   schunk.SchunkPZV64,
}


def main():
    """
    Visualize a UR10e robot with multiple tools in Rerun.
    """

    # Create robot
    robot = universal_robots.UniversalRobotsUR10E()

    # Move the robot to an end-effector-down orientation
    robot.set_joint_positions([180, -90, 90, -90, -90, 0])

    # Attach and visualize each supported tool in turn
    for gripper_cls in _GRIPPERS.values():
        gripper = gripper_cls()
        robot.attach_tool(gripper)
        robot.visualize_rerun()
        time.sleep(1)  # Allow Rerun to update the visualization
        logger.info(f"Visualizing {gripper_cls.__name__}...")

if __name__ == "__main__":
    main()
