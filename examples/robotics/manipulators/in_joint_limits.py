"""
Checks whether joint configurations lie within the limits derived from the robot's URDF.

Supports Universal Robots (UR), Epson, virtual, and Isaac Sim.

Usage:
    python in_joint_limits.py [--ip <ROBOT_IP>] [--prim_path <PRIM_PATH>]
"""

import argparse

from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots


def main() -> None:
    """Check the robot's current joint configuration and an out-of-range one."""

    # ===================== Create Robot ==========================================
    robot = universal_robots.UniversalRobotsUR10E(name='UR10e')

    try:
        # ==================== Run Skill ============================================
        current_joint_positions = robot.get_joint_positions()
        logger.success(
            f"Current joint positions within limits: "
            f"{robot.in_joint_limits(q=current_joint_positions, verbose=True)}"
        )

        out_of_range = robot.joint_limits[:, 1] + 10.0  # 10 deg past every upper limit
        logger.info(
            f"Configuration past the upper limits within limits: "
            f"{robot.in_joint_limits(q=out_of_range, verbose=True)}"
        )
    except (RuntimeError, TypeError, ValueError) as e:
        logger.error(f"IK failed: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
