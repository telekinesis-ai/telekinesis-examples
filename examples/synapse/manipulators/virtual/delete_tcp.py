"""
Register a TCP frame, then delete it.

Supports Universal Robots (UR), Epson, and virtual/sim.

Usage:
    python delete_tcp.py
"""

from loguru import logger

from telekinesis.synapse.robots.manipulators.universal_robots import UniversalRobotsUR10E


def main():
    """Add a TCP, then delete it, observing the active TCP at each step."""

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

    # Delete the TCP
    robot.delete_tcp(name="new_tool")

    # Active TCP, transform w.r.t default tcp, and TCP pose
    logger.info(f"Active TCP after delete_tcp(): {robot.active_tcp}"
                f" \nActive TCP transform: {robot.get_active_tcp_transform()}"
                f" \n TCP pose: {robot.get_cartesian_pose()}")


if __name__ == "__main__":
    main()
