"""
Register a custom TCP frame and inspect the active TCP before and after.

Supports Universal Robots (UR), Epson, and virtual/sim.

Usage:
    python add_tcp.py
"""

from loguru import logger

from telekinesis.synapse.robots.manipulators.universal_robots import UniversalRobotsUR10E


def main():
    """Observe the active TCP and its transform before and after add_tcp()."""

    #===================== Create Robot ==========================================
    robot = UniversalRobotsUR10E(name='UR10e')

    # ==================== Visualization (Optional) =============================
    robot.visualize_rerun(live=True)
    
    # ==================== Run Skill ============================================
    # Current Active TCP, transform w.r.t default tcp, and current TCP pose
    logger.info(f"Active TCP before add_tcp(): {robot.active_tcp}"
                f" \nActive TCP transform: {robot.get_active_tcp_transform()}"
                f" \n TCP pose: {robot.get_cartesian_pose()}")

    # Add new tcp
    robot.add_tcp(name="new_tool",
                  transform=[0.0, 0.0, 0.1, 0.0, 0.0, 0.0],  # 100 mm along Z-axis
                  set_active=True)

    # Get updated Active TCP, transform w.r.t default tcp, and TCP pose
    logger.info(f"Active TCP after add_tcp(): {robot.active_tcp}"
                f" \nActive TCP transform: {robot.get_active_tcp_transform()}"
                f" \n TCP pose: {robot.get_cartesian_pose()}")

    


if __name__ == "__main__":
    main()
