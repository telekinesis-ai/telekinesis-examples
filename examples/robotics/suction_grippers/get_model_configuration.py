"""
Demonstrates how to read the URDF joint configuration of a suction gripper
kinematics model for the Synapse SDK.

Supports Piab grippers.

Usage:
    python get_model_configuration.py
"""

from loguru import logger

from telekinesis.synapse.tools.suction_grippers import piab


def main() -> None:
    """Reads the model configuration of a Piab gripper."""

    #===================== Create Gripper ======================================
    gripper = piab.PiabPiCobotElectric()

    # ==================== Run Skill ===========================================
    q = gripper.get_model_configuration()
    logger.success(f"Model configuration (URDF units): {q} (shape {q.shape})")


if __name__ == "__main__":
    main()
