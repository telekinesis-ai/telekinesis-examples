"""
Attach every supported parallel gripper to a manipulator in turn and visualize
the result.

Supports Universal Robots (UR), Epson, and virtual.

Usage:
    python attach_and_visualize_all_tools_supported.py
"""

import time
from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots
from telekinesis.synapse.tools.parallel_grippers import robotiq, onrobot, schunk

# Every parallel gripper model the SDK ships, keyed by short name.
_GRIPPERS = {
    "2f85":    robotiq.Robotiq2F85,
    "2f140":   robotiq.Robotiq2F140,
    "hande":   robotiq.RobotiqHandE,
    "rg6":     onrobot.OnRobotRG6,
    "rg2":     onrobot.OnRobotRG2,
    "egu50":   schunk.SchunkEGU50,
    "egp":     schunk.SchunkEGP,
    "pznplus": schunk.SchunkPZNPlus,
    "pzv64":   schunk.SchunkPZV64,
}

# Pause between grippers, so each one is visible in the viewer before the
# next replaces it.
DELAY_BETWEEN_TOOLS_S = 1.0


def main():
    """Attaches each supported parallel gripper to a UR10e and visualizes it."""

    #===================== Create Robot =========================================
    # The name is what enables live updates: it gives the robot a state topic
    # for visualize_rerun(live=True) to subscribe to.
    robot = universal_robots.UniversalRobotsUR10E(name='UR10e')

    # ==================== Visualization (Optional) =============================
    # Called once, before the loop: the viewer subscribes to the robot's state
    # topic and redraws by itself on every later change, so the attach calls
    # below need no further visualize_rerun() call.
    robot.visualize_rerun(live=True)

    # ==================== Run Skill ============================================
    # Move the robot to an end-effector-down orientation
    robot.set_joint_positions([180, -90, 90, -90, -90, 0])

    # Attach and visualize each supported tool in turn
    for name, gripper_cls in _GRIPPERS.items():
        gripper = gripper_cls()

        # Clear the previous gripper's meshes, so the tools do not pile up in
        # the viewer as the loop goes on.
        robot.detach_tool()
        robot.attach_tool(gripper)

        logger.info(f"Visualizing {name}: {gripper_cls.__name__}...")
        time.sleep(DELAY_BETWEEN_TOOLS_S)

    logger.success(f"Visualized {len(_GRIPPERS)} supported parallel grippers.")


if __name__ == "__main__":
    main()
