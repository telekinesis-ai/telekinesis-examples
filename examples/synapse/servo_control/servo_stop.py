"""
Servo Stop example for the Synapse SDK.

Starts a brief ``servo_joint`` streaming move and then interrupts it with
``servo_stop``. ``deceleration`` controls how quickly the controller ramps
the joints down [deg/s²].

Currently supported only for Universal Robots (UR10e).

Usage:
    python servo_stop.py --ip <ROBOT_IP>
"""

import argparse
import math
import time

from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots


def main(robot_ip: str):
    """Stream servo_joint targets for 1 second, then interrupt with servo_stop."""

    # Servo-loop and trajectory parameters
    dt = 0.008          # 125 Hz servo loop
    amplitude = 2.0     # ±2 deg base oscillation
    period = 4.0        # seconds per full oscillation
    stream_duration = 1.0  # stream servo targets for this long before stopping
    deceleration = 10.0    # deg/s² for servo_stop

    # Create robot instance
    robot = universal_robots.UniversalRobotsUR10E()

    # Connect to the robot
    robot.connect(ip=robot_ip)

    try:
        # Hold all joints at their current values; only j0 will be modulated.
        center = robot.get_joint_positions()
        logger.info(
            f"Streaming j0 oscillation for {stream_duration}s, then servo_stop"
        )

        t0 = time.monotonic()
        while True:
            t = time.monotonic() - t0
            if t >= stream_duration:
                break

            theta = 2.0 * math.pi * t / period
            target = list(center)
            target[0] = center[0] + amplitude * math.sin(theta)

            robot.servo_joint(
                q=target,
                speed=60.0,
                acceleration=80.0,
                time=dt,
                lookahead_time=0.1,
                gain=300,
            )

            # Pace the loop. Sleep the remainder of this dt window.
            next_tick = t0 + (math.floor(t / dt) + 1) * dt
            sleep_for = next_tick - time.monotonic()
            if sleep_for > 0:
                time.sleep(sleep_for)

        # Interrupt the servo stream — controller ramps the joints down.
        robot.servo_stop(deceleration=deceleration)
        logger.success(f"servo_stop issued (deceleration={deceleration} deg/s²).")
    finally:
        robot.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UR10e servo_stop example")
    parser.add_argument("--ip", type=str, required=True, help="IP address of the UR robot")
    args = parser.parse_args()

    main(args.ip)
