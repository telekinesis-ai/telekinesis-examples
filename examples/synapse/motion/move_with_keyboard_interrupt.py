"""
Move a real UR10e to a Cartesian target with a clean Ctrl+C stop.

The pose is sent asynchronously; a polling loop waits for steady state so the
move completes before exit. Pressing Ctrl+C calls ``stop_cartesian_motion()``.

Supports only Universal Robots hardware.

Usage:
    python move_with_keyboard_interrupt.py [--ip <ROBOT_IP>]
"""

import argparse
import time

from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots


def main(ip: str) -> None:
    robot = universal_robots.UniversalRobotsUR10E()
    robot.connect(ip=ip)

    target_cartesian_pose = [0.40, 0.10, 0.45, 180.0, 0.0, 0.0]

    try:
        logger.info("Moving to {} -- press Ctrl+C to stop cleanly.", target_cartesian_pose)
        robot.set_cartesian_pose(target_cartesian_pose, speed=1.05, acceleration=1.4, asynchronous=True)

        # Wait for the async move to finish (Ctrl+C interrupts).
        while not robot.is_steady():
            time.sleep(0.01)

        logger.success(f"Reached: {robot.get_cartesian_pose()}")

    except KeyboardInterrupt:
        logger.warning("Interrupted -- stopping robot.")
        robot.stop_cartesian_motion(stopping_speed=0.5)

    finally:
        robot.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Move UR10e to a Cartesian target; Ctrl+C stops cleanly.")
    parser.add_argument("--ip", default="192.168.1.100", help="UR robot IP address (default: 192.168.1.100)")
    args = parser.parse_args()
    main(ip=args.ip)
