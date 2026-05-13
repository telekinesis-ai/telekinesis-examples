"""
Servo Cartesian example for the Synapse SDK.

Streams TCP poses at 500 Hz to trace a small circle in the YZ plane around
the current TCP pose using ``servo_cartesian``.

Currently supported only for Universal Robots (UR10e).

Usage:
    python servo_cartesian.py --ip <ROBOT_IP>
"""

import argparse
import math
import time

from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots

def main(robot_ip: str):
    """Trace a YZ circle around the current TCP pose with servo_cartesian."""

    dt = 0.002              # 500 Hz servo loop
    radius = 0.02           # 2 cm circle
    period = 4.0            # seconds per revolution
    n_revolutions = 2

    # Create robot instance
    robot = universal_robots.UniversalRobotsUR10E()

    # Connect to the robot
    robot.connect(ip=robot_ip)

    try:
        # Read the current TCP pose as the centre of the circle.
        # The circle is offset so it "kisses" the start pose at t=0.
        center = robot.get_cartesian_pose()
        logger.info(f"Tracing YZ circle (r={radius} m) around {center}")

        duration = period * n_revolutions
        t0 = time.monotonic()
        while True:
            t = time.monotonic() - t0
            if t >= duration:
                break

            theta = 2.0 * math.pi * t / period
            target = center[:]
            target[1] = center[1] + radius * math.cos(theta) - radius
            target[2] = center[2] + radius * math.sin(theta)

            robot.servo_cartesian(
                pose=target,
                speed=0.1,
                acceleration=0.1,
                time=dt,
                lookahead_time=0.1,
                gain=300,
            )

            # Pace the loop. Sleep the remainder of this dt window.
            next_tick = t0 + (math.floor(t / dt) + 1) * dt
            sleep_for = next_tick - time.monotonic()
            if sleep_for > 0:
                time.sleep(sleep_for)

        robot.servo_stop()
        logger.success("servo_cartesian loop complete.")
    finally:
        robot.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UR10e servo_cartesian example")
    parser.add_argument("--ip", type=str, required=True, help="IP address of the UR robot")
    args = parser.parse_args()

    main(args.ip)
