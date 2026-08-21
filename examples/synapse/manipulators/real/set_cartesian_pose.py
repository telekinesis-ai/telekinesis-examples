"""
Moves the TCP to a target Cartesian pose relative to its current pose.

Supports Universal Robots (UR), Epson, and virtual/sim.

Usage:
    python set_cartesian_pose.py [--ip <ROBOT_IP>]
"""

import argparse

from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots


def main(ip: str) -> None:
    """Move the TCP to a target Cartesian pose relative to its current pose."""

    #===================== Create Robot ==========================================
    robot = universal_robots.UniversalRobotsUR10E(name='UR10e')

    # ==================== Visualization (Optional) =============================
    # Live: subscribes to the robot's state topic and redraws as it moves.
    robot.visualize_rerun(live=True)

    try:
        #===================== Connect Robot ==========================================
        robot.connect(ip=ip)

        #===================== Prepare Target ==========================================
        current_cartesian_pose = robot.get_cartesian_pose()
        target_cartesian_pose = current_cartesian_pose.copy()
        target_cartesian_pose[2] += 0.1  # Move 10 cm up in Z

        # ==================== Run Skill ============================================
        robot.set_cartesian_pose(
            cartesian_pose=target_cartesian_pose,
            speed=0.5,
            acceleration=0.5,
        )
        logger.info(f"Moved to target Cartesian pose: {target_cartesian_pose}")
    except (ConnectionError, OSError) as e:
        logger.error(f"Error occurred: {e}")
    finally:
        robot.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Move the TCP to a target Cartesian pose")
    parser.add_argument("--ip", type=str, default="192.168.1.100", help="UR robot IP address (default: 192.168.1.100)")
    args = parser.parse_args()

    main(ip=args.ip)

