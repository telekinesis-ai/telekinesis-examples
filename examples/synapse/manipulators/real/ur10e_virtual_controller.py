"""
Connects to a URSim virtual controller, reads robot state, and moves to the default joint configuration.

Supports Universal Robots (UR), Epson, and virtual/sim.

Prerequisites:
  - URSim running in Docker with ports 30001-30004 and 29999 exposed
  - Remote Control enabled in the URSim teach pendant
    (hamburger menu -> Settings -> System -> Remote Control -> Enable)

Usage:
    python ur10e_virtual_controller.py [--ip <ROBOT_IP>]
"""

import argparse

from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots


def main(ip: str) -> None:
    """Connect to a URSim virtual controller and control a UR10e."""

    #===================== Create Robot ==========================================
    robot = universal_robots.UniversalRobotsUR10E(name='UR10e')

    # ==================== Visualization (Optional) =============================
    robot.visualize_rerun(live=True)

    try:
        #===================== Connect Robot ==========================================
        robot.connect(ip=ip)

        # ==================== Run Skill ============================================
        logger.info(f"Robot mode:   {robot.get_robot_mode()}")
        logger.info(f"Safety mode:  {robot.get_safety_mode()}")
        logger.info(f"Robot status: {robot.get_robot_status()}")
        logger.info(f"Joint positions (deg): {robot.get_joint_positions()}")
        logger.info(f"TCP pose (m, deg):     {robot.get_cartesian_pose()}")

        # Move command
        robot.set_joint_positions(robot.default_joint_configuration)
    except (ConnectionError, OSError) as e:
        logger.error(f"Error occurred: {e}")
    finally:
        robot.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Connect to a URSim virtual controller and control a UR10e")
    parser.add_argument("--ip", type=str, default="127.0.0.1", help="URSim virtual controller IP address (default: 127.0.0.1)")
    args = parser.parse_args()

    main(ip=args.ip)
