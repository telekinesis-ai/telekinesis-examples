"""
Register multiple TCP frames and switch the active one.

Supports Universal Robots (UR), Epson, and virtual.

Usage:
    python change_active_tcp.py
"""

from loguru import logger

from telekinesis.synapse.robots.manipulators.universal_robots import UniversalRobotsUR10E


def main():
    """Change the active TCP and observe it before and after each change."""

    #===================== Create Robot ==========================================
    robot = UniversalRobotsUR10E(name='UR10e')

    # ==================== Visualization (Optional) =============================
    robot.visualize_rerun(live=True)

    # ==================== Run Skill ============================================
    # Register a few custom TCP frames
    robot.add_tcp(name="camera_tip",
                  transform=[0.2, 0.2, 0.1, 0.0, 0.0, 90.0],
                  set_active=True)
    robot.add_tcp(name="gripper_tip",
                  transform=[0.0, 0.0, 0.26, 0.0, 0.0, 180.0],
                  set_active=False)
    robot.add_tcp(name="laser_tip",
                  transform=[-0.05, 0.0, 0.15, 0.0, 0.0, 90.0],
                  set_active=False)

    # Active TCP, transform w.r.t default tcp, and TCP pose
    logger.info(f"Active TCP after add_tcp(): {robot.active_tcp}"
                f" \nActive TCP transform: {robot.get_active_tcp_transform()}"
                f" \n TCP pose: {robot.get_cartesian_pose()}")

    # Change the active TCP
    robot.active_tcp = "gripper_tip"
    logger.info(f"Active TCP after changing active TCP: {robot.active_tcp}"
                f" \nActive TCP transform: {robot.get_active_tcp_transform()}"
                f" \n TCP pose: {robot.get_cartesian_pose()}")

    # Change the active TCP again
    robot.active_tcp = "laser_tip"
    logger.info(f"Active TCP after changing active TCP again: {robot.active_tcp}"
                f" \nActive TCP transform: {robot.get_active_tcp_transform()}"
                f" \n TCP pose: {robot.get_cartesian_pose()}")


if __name__ == "__main__":
    main()
