"""
Register a TCP frame, then update its transform.

Supports Universal Robots (UR), Epson, and virtual.

Usage:
    python update_tcp.py
"""

from loguru import logger

from telekinesis.synapse.robots.manipulators.universal_robots import UniversalRobotsUR10E


def main():
    """Add a TCP, then update its transform, observing it at each step."""

    #===================== Create Robot ==========================================
    robot = UniversalRobotsUR10E(name='UR10e')

    # ==================== Visualization (Optional) =============================
    robot.visualize_rerun(live=True)

    # ==================== Run Skill ============================================
    robot.add_tcp(name="new_tool",
                  transform=[0.0, 0.0, 0.1, 0.0, 0.0, 0.0],  # 100 mm along Z-axis
                  set_active=True)

    # Active TCP, transform w.r.t default tcp, and TCP pose
    logger.info(f"Active TCP after add_tcp(): {robot.active_tcp}"
                f" \nActive TCP transform: {robot.get_active_tcp_transform()}"
                f" \n TCP pose: {robot.get_cartesian_pose()}")

    # Update the TCP
    updated_tcp_pose_in_default_tcp_frame = [0.0, 0.0, 0.2, 0.0, 0.0, 0.0]  # 200 mm along Z-axis
    robot.update_tcp(name="new_tool",
                     transform=updated_tcp_pose_in_default_tcp_frame)

    # Active TCP, transform w.r.t default tcp, and TCP pose
    logger.info(f"Active TCP after update_tcp(): {robot.active_tcp}"
                f" \nActive TCP transform: {robot.get_active_tcp_transform()}"
                f" \n TCP pose: {robot.get_cartesian_pose()}")


if __name__ == "__main__":
    main()
