"""
Commands an asynchronous joint move and interrupts it mid-trajectory with stop_joint_motion.

Supports Universal Robots (UR), Epson, and virtual/sim.

Usage:
    python stop_joint_motion.py [--ip <ROBOT_IP>]
"""

import argparse
import time

from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots


def main(ip: str) -> None:
    """Start an async joint move and interrupt it with stop_joint_motion."""

    #===================== Create Robot ==========================================
    robot = universal_robots.UniversalRobotsUR10E(name='UR10e')

    # ==================== Visualization (Optional) =============================
    # Live: subscribes to the robot's state topic and redraws as it moves.
    robot.visualize_rerun(live=True)

    try:
        #===================== Connect Robot ==========================================
        robot.connect(ip=ip)

        #===================== Prepare Target ==========================================
        # Get initial joint positions [deg]
        initial_joint_positions = robot.get_joint_positions()
        target_joint_positions = list(initial_joint_positions)
        target_joint_positions[0] += 20  # Asynchronous +20 deg move on joint 0

        # ==================== Run Skill ============================================
        robot.set_joint_positions(
            joint_positions=target_joint_positions,
            speed=60,
            acceleration=80,
            asynchronous=True,
        )

        # Let the move run briefly, then interrupt it
        time.sleep(0.3)
        robot.stop_joint_motion(stopping_speed=30)
        logger.info("Stopped joint motion.")
    except (ConnectionError, OSError) as e:
        logger.error(f"Error occurred: {e}")
    finally:
        robot.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interrupt an async joint move with stop_joint_motion")
    parser.add_argument("--ip", type=str, default="192.168.1.100", help="UR robot IP address (default: 192.168.1.100)")
    args = parser.parse_args()

    main(ip=args.ip)
