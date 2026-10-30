"""
Demonstrates limiting how far a parallel gripper can be commanded to open.

Supports all.

Usage:
    python set_max_opening_width.py
"""

from loguru import logger

from telekinesis.synapse.tools.parallel_grippers import onrobot


def main() -> None:
    """Limits an OnRobot gripper to a 40 mm maximum opening."""

    #===================== Create Gripper ======================================
    gripper = onrobot.OnRobotRG6()

    # ==================== Run Skill ===========================================
    logger.info(f"Physical maximum opening: {gripper.physical_max_opening_width_mm} mm")
    gripper.set_max_opening_width_mm(40.0)
    logger.success("Maximum commanded opening set to 40.0 mm.")


if __name__ == "__main__":
    main()
