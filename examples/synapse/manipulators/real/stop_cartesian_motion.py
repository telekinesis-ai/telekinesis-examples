"""
Commands an asynchronous Cartesian move and interrupts it mid-trajectory with stop_cartesian_motion.

Supports Universal Robots (UR).

Usage:
    python stop_cartesian_motion.py [--ip <ROBOT_IP>]
"""

import argparse
import time

from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots


def main(ip: str) -> None:
    """Start an async Cartesian move and interrupt it with stop_cartesian_motion."""

    #===================== Create Robot ==========================================
    robot = universal_robots.UniversalRobotsUR10E(name='UR10e')

    # ==================== Visualization (Optional) =============================
    # Live: subscribes to the robot's state topic and redraws as it moves.
    robot.visualize_rerun(live=True)

    try:
        #===================== Connect Robot ==========================================
        robot.connect(ip=ip)

        #===================== Prepare Target ==========================================
        # Get initial Cartesian pose [x, y, z, rx, ry, rz] (m, deg)
        actual_pose = robot.get_cartesian_pose()
        target_pose = list(actual_pose)
        target_pose[2] += 0.15  # Asynchronous +15 cm move along Z

        # ==================== Run Skill ============================================
        robot.set_cartesian_pose(
            cartesian_pose=target_pose,
            speed=0.25,
            acceleration=0.5,
            asynchronous=True,
        )

        # Let the move run briefly, then interrupt it
        time.sleep(0.3)
        robot.stop_cartesian_motion(stopping_speed=0.25)
        logger.info("Stopped Cartesian motion.")
    except (ConnectionError, OSError) as e:
        logger.error(f"Error occurred: {e}")
    finally:
        robot.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interrupt an async Cartesian move with stop_cartesian_motion")
    parser.add_argument("--ip", type=str, default="192.168.1.100", help="UR robot IP address (default: 192.168.1.100)")
    args = parser.parse_args()

    main(ip=args.ip)
