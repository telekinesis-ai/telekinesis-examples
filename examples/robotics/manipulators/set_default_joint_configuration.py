"""
Override the default joint configuration used as the offline commanded state.

Supports Universal Robots (UR), Epson, and virtual.

Usage:
    python set_default_joint_configuration.py
"""

from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots


def main():
    """Override the default joint configuration and verify the readback."""

    #===================== Create Robot ==========================================
    robot = universal_robots.UniversalRobotsUR10E(name='UR10e')

    # ==================== Run Skill ============================================
    # Log the brand-default joint configuration [deg]
    logger.info(f"Default joint configuration [deg]: {robot.default_joint_configuration}")

    # Set a new default joint configuration [deg]
    new_default = [0.0, -90.0, -90.0, 0.0, 90.0, 0.0]
    robot.set_default_joint_configuration(q=new_default)
    logger.success(f"Updated default joint configuration [deg]: {robot.default_joint_configuration}")

    # ==================== Visualization (Optional) =============================
    # set_default_joint_configuration() only changes the IK seed, not the
    # commanded state — drive the robot there to see the change reflected.
    robot.set_joint_positions(joint_positions=new_default)
    robot.visualize_rerun(live=False)


if __name__ == "__main__":
    main()
