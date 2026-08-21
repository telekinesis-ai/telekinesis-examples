"""
Drives the TCP slowly downward while polling contact detection, and stops as soon as the tool touches a surface.

Supports Universal Robots (UR)

Usage:
    python contact_detection.py [--ip <ROBOT_IP>]
"""

import argparse
import time

from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots


def main(ip: str) -> None:
    """Probe downward until contact is detected, then stop and report."""

    #===================== Create Robot ==========================================
    robot = universal_robots.UniversalRobotsUR10E(name='UR10e')

    try:
        #===================== Connect Robot ==========================================
        robot.connect(ip=ip)

        #===================== Prepare Target ==========================================
        target_pose = robot.get_cartesian_pose()
        target_pose[2] -= 0.15  # Move 15 cm down in Z

        # ==================== Run Skill ============================================
        robot.start_contact_detection()
        robot.set_cartesian_pose(cartesian_pose=target_pose, speed=0.05,
                                 acceleration=0.25, asynchronous=True)

        # Poll until contact, or for up to 5 s if nothing is hit.
        contact = False
        deadline = time.time() + 5.0
        while not contact and time.time() < deadline:
            contact = robot.read_contact_detection()
            time.sleep(0.02)

        robot.stop_cartesian_motion(stopping_speed=0.25)
        robot.stop_contact_detection()
        logger.success(f"Contact: {contact}")
    except (ConnectionError, OSError) as e:
        logger.error(f"Error occurred: {e}")
    finally:
        robot.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Probe downward with contact detection polling (start/read/stop)")
    parser.add_argument("--ip", type=str, default="192.168.1.100", help="UR robot IP address (default: 192.168.1.100)")
    args = parser.parse_args()

    main(ip=args.ip)
