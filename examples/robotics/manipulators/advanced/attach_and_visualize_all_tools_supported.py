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
from telekinesis.synapse.tools.suction_grippers import piab

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
    "piabpicobot": piab.PiabPiCobotElectric,
}

# Pause between grippers, so each one is visible in the viewer before the
# next replaces it.
DELAY_BETWEEN_TOOLS_S = 1.0


def main():
    """Attaches each supported parallel gripper to a UR10e and visualizes it."""

    #===================== Create Robot =========================================
    robot = universal_robots.UniversalRobotsUR10E(name='UR10e')

    # ==================== Visualization (Optional) =============================
    robot.visualize_rerun(live=True)

    # ==================== Run Skill ============================================
    # Move the robot to an end-effector-down orientation
    robot.set_joint_positions([180, -90, 90, -90, -90, 0])

    # Attach and visualize each supported tool in turn
    for name, gripper_cls in _GRIPPERS.items():
        gripper = gripper_cls()

        robot.detach_tool()
        robot.attach_tool(gripper)

        time.sleep(DELAY_BETWEEN_TOOLS_S)

    # =================== Shutdown ===============================================
    robot.shutdown()


if __name__ == "__main__":
    main()
