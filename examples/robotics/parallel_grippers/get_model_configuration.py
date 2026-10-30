"""
Demonstrates how to read the URDF joint configuration of a gripper kinematics
model for the Synapse SDK.

Supports all.

Usage:
    python get_model_configuration.py
"""

from loguru import logger

from telekinesis.synapse.tools.parallel_grippers import onrobot


def main() -> None:
    """Reads the model configuration of an OnRobot gripper, then queries other openings."""

    #===================== Create Gripper ======================================
    gripper = onrobot.OnRobotRG6()

    # ==================== Run Skill ===========================================
    q = gripper.get_model_configuration()
    logger.success(f"Model configuration (URDF units): {q}")

    # The same model queried at other openings, without changing the gripper.
    q_half = gripper.get_model_configuration_at_joint_travel_fraction(0.5)
    q_30mm = gripper.get_model_configuration_at_opening_width(30.0, unit="mm")
    logger.info(f"Halfway through joint travel: {q_half}")
    logger.info(f"At a 30 mm opening: {q_30mm}")


if __name__ == "__main__":
    main()
