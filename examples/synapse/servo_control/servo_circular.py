"""
Servo Circular example for the Synapse SDK.

Commands a circular-arc move from the current TCP pose to a target pose
using ``servo_circular`` (UR ``servoC``). The target here is offset 2 cm
along Z and 2 cm along Y so the arc is visually distinct from a straight
line.

Currently supported only for Universal Robots (UR10e).
Usage:
    python servo_circular.py --ip <ROBOT_IP>
"""

import argparse
import time

from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots


def main(robot_ip: str):
    """Drive a circular arc from the current TCP pose to an offset target."""

    # Create robot instance
    robot = universal_robots.UniversalRobotsUR10E()

    # Connect to the robot
    robot.connect(ip=robot_ip)

    try:
        # Target pose: 2 cm down in Z and 2 cm out in Y from the current pose.
        current = robot.get_cartesian_pose()
        target = list(current)
        target[1] += 0.02
        target[2] -= 0.02

        logger.warning(
            f"About to move real robot along a circular arc from {current} to {target}. "
            "Make sure it's safe to move there."
        )
        logger.info(f"servo_circular target: {target}")

        # Command the circular servo move, then stop streaming to end the motion.
        robot.servo_circular(
            pose=target,
            speed=0.1,
            acceleration=0.1,
            blend=0.0,
        )

        time.sleep(2.0)  # In a real application, you would typically stream continuously until some condition is met (e.g. a certain time has elapsed, or a sensor triggers).

        # In a real application, you would typically stream continuously until some
        # condition is met (e.g. a certain time has elapsed, or a sensor triggers).
        robot.servo_stop()
        logger.success("servo_circular complete.")
    finally:
        robot.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UR10e servo_circular example")
    parser.add_argument("--ip", type=str, required=True, help="IP address of the UR robot")
    args = parser.parse_args()

    main(args.ip)
